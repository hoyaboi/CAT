"""
Word LLM generator for creating word substitution dictionaries.
"""
import yaml
import os
import re
from pathlib import Path
from typing import Optional, Dict, List, Set
import nltk
from nltk.tag import pos_tag

from model import ModelFactory, LLMClient
from src.word import Dictionary

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    try:
        nltk.download('averaged_perceptron_tagger', quiet=True)
    except:
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)


WORD_PROMPT_PATH = Path("prompts/word_llm_system.yaml")
MAX_ATTEMPTS = 5
VALID_CATEGORIES = {'noun', 'verb', 'adjective', 'adverb', 'unit'}


def _extract_keywords_from_query(query: str) -> Dict[str, Set[str]]:
    """
    Extract key words from harmful query by POS category.
    
    Args:
        query: Harmful query string
    
    Returns:
        Dictionary mapping category to set of keywords
    """
    keywords = {
        'noun': set(),
        'verb': set(),
        'adjective': set(),
        'adverb': set(),
        'unit': set()
    }
    
    # Tokenize and tag
    words = nltk.word_tokenize(query.lower())
    pos_tags = pos_tag(words)
    
    # Common stop words to exclude
    stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                  'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 
                  'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 
                  'should', 'could', 'may', 'might', 'can', 'must', 'this', 'that', 
                  'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 
                  'how', 'what', 'when', 'where', 'why', 'who', 'which', 'give', 
                  'detailed', 'instructions', 'on', 'make', 'other'}
    
    # Map POS tags to categories
    for word, tag in pos_tags:
        # Remove punctuation
        word_clean = re.sub(r'[^\w]', '', word.lower())
        if not word_clean or word_clean in stop_words or len(word_clean) < 2:
            continue
        
        # Map POS tags to our categories
        if tag.startswith('NN'):  # Noun
            keywords['noun'].add(word_clean)
        elif tag.startswith('VB'):  # Verb
            keywords['verb'].add(word_clean)
        elif tag.startswith('JJ'):  # Adjective
            keywords['adjective'].add(word_clean)
        elif tag.startswith('RB'):  # Adverb
            keywords['adverb'].add(word_clean)
    
    return keywords


def generate_dictionary(
    harmful_query: str,
    target_category: str,
    word_llm_client: Optional[LLMClient] = None,
    output_dir: str = "results/dictionaries",
    task_num: Optional[int] = None
) -> Dictionary:
    """Generate word substitution dictionary using Word LLM."""
    if word_llm_client is None:
        word_llm_client = ModelFactory.create_word_llm()
    
    # Extract key words from harmful query
    key_words = _extract_keywords_from_query(harmful_query)
    
    harmful_words = _generate_word_list(
        context=harmful_query,
        word_llm_client=word_llm_client,
        task_num=task_num,
        output_dir=output_dir,
        list_type="harmful",
        key_words=key_words
    )
    
    benign_words = _generate_word_list(
        context=target_category,
        word_llm_client=word_llm_client,
        task_num=task_num,
        output_dir=output_dir,
        list_type="benign"
    )
    
    all_mappings = _match_word_lists(harmful_words, benign_words)
    
    dictionary = Dictionary(all_mappings)
    _validate_and_save_dictionary(dictionary, target_category, output_dir, task_num)
    
    return dictionary


def _generate_word_list(
    context: str,
    word_llm_client: LLMClient,
    task_num: Optional[int],
    output_dir: str,
    list_type: str,
    key_words: Optional[Dict[str, Set[str]]] = None
) -> Dict[str, List[str]]:
    """Generate a list of words (harmful or benign) organized by category."""
    all_words: Dict[str, List[str]] = {}
    
    for category, expected_count in Dictionary.EXPECTED_COUNTS.items():
        category_keywords = key_words.get(category, set()) if key_words else set()
        category_words = _generate_category_words_with_retry(
            context=context,
            category=category,
            expected_count=expected_count,
            word_llm_client=word_llm_client,
            task_num=task_num,
            output_dir=output_dir,
            list_type=list_type,
            key_words=category_keywords
        )
        all_words[category] = category_words
    
    return all_words


def _generate_category_words_with_retry(
    context: str,
    category: str,
    expected_count: int,
    word_llm_client: LLMClient,
    task_num: Optional[int],
    output_dir: str,
    list_type: str,
    key_words: Optional[Set[str]] = None
) -> List[str]:
    """Generate words for a category by generating 2x expected_count words at a time until we have enough."""
    category_words = []
    seen_words = set()
    attempt = 0
    
    # First, add key words at the beginning to ensure they're included in the mapping
    if key_words and list_type == "harmful":
        for key_word in sorted(key_words):  # Sort for consistency
            key_word_lower = key_word.lower()
            if key_word_lower not in seen_words:
                seen_words.add(key_word_lower)
                category_words.append(key_word)
                if len(category_words) >= expected_count:
                    return category_words
    
    while len(category_words) < expected_count and attempt < MAX_ATTEMPTS:
        attempt += 1
        
        # Calculate how many words to generate: 2x the expected count
        words_to_generate = expected_count * 2
        
        response = _call_llm_for_words(
            context=context,
            category=category,
            existing_words=set(category_words),
            word_llm_client=word_llm_client,
            task_num=task_num,
            output_dir=output_dir,
            list_type=list_type,
            key_words=key_words,
            word_count=words_to_generate
        )
        
        parsed_words = _parse_word_list(response, category)
        
        for word in parsed_words:
            if len(category_words) >= expected_count:
                break
            word_lower = word.lower()
            if word_lower not in seen_words:
                seen_words.add(word_lower)
                category_words.append(word)
        
        # If we have enough words, return early
        if len(category_words) >= expected_count:
            return category_words
    
    return category_words


def _call_llm_for_words(
    context: str,
    category: str,
    existing_words: set,
    word_llm_client: LLMClient,
    task_num: Optional[int],
    output_dir: str,
    list_type: str,
    key_words: Optional[Set[str]] = None,
    word_count: int = 100
) -> str:
    """Call LLM to generate words for a category."""
    system_prompt = _load_and_prepare_prompt(category, list_type, word_count)
    user_prompt = _create_user_prompt(context, category, existing_words, list_type, key_words)
    
    response = word_llm_client.call(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.7,
        max_tokens=800,
    )
    
    return response


def _load_and_prepare_prompt(category: str, list_type: str, word_count: int = 100) -> str:
    """Load prompt template and replace placeholders."""
    if not WORD_PROMPT_PATH.exists():
        raise FileNotFoundError(f"Prompt file not found: {WORD_PROMPT_PATH}")
    
    with open(WORD_PROMPT_PATH, 'r', encoding='utf-8') as f:
        prompt_config = yaml.safe_load(f)
        template = prompt_config['prompt']
    
    # Replace placeholders
    template = template.replace("[CATEGORY]", category)
    template = template.replace("[WORD_COUNT]", str(word_count))
    
    return template


def _create_user_prompt(
    context: str,
    category: str,
    existing_words: set,
    list_type: str,
    key_words: Optional[Set[str]] = None
) -> str:
    """Create user prompt based on list type."""
    existing_list = ", ".join(list(existing_words)[:20]) if existing_words else "None"
    
    if list_type == "harmful":
        # Build simple user prompt with just essential information
        prompt_parts = [context]
        
        # Add key words if available
        if key_words and len(key_words) > 0:
            key_words_list = ", ".join(sorted(list(key_words))[:10])
            prompt_parts.append(f"\nKey words to include: {key_words_list}")
            prompt_parts.append(f"\nNote: Only include words that match the {category} part of speech.")
        
        # Add existing words if any
        if existing_words:
            prompt_parts.append(f"\nAlready generated: {existing_list}")
        
        return "\n".join(prompt_parts)
    else:
        # Build simple user prompt for benign words
        prompt_parts = [context]
        
        # Add existing words if any
        if existing_words:
            prompt_parts.append(f"\nAlready generated: {existing_list}")
        
        return "\n".join(prompt_parts)


def _parse_word_list(response: str, category: str) -> List[str]:
    """Parse word list from LLM response."""
    words = []
    seen_words = set()
    lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
    
    for line in lines:
        word = _extract_word_from_line(line)
        if word:
            word_lower = word.lower()
            if word_lower not in seen_words:
                seen_words.add(word_lower)
                words.append(word)
    
    return words


def _extract_word_from_line(line: str) -> Optional[str]:
    """Extract a single word from a line."""
    if not line or ' ' in line:
        return None
    
    line_lower = line.lower().strip()
    if line_lower.startswith(('category', 'word', 'output', 'format', 'example', 'requirements')):
        return None
    
    if ':' in line:
        return None
    
    word = line.strip().rstrip('.,;:!?')
    
    if not word or not word.replace('-', '').replace('_', '').isalnum():
        return None
    
    invalid_chars = ['(', ')', '[', ']', '{', '}', '=', '+', '*', '/', '\\']
    if any(char in word for char in invalid_chars):
        return None
    
    return word


def _match_word_lists(
    harmful_words: Dict[str, List[str]],
    benign_words: Dict[str, List[str]]
) -> Dict[str, Dict[str, str]]:
    """Match harmful and benign word lists 1:1 by category."""
    all_mappings: Dict[str, Dict[str, str]] = {}
    
    for category in Dictionary.EXPECTED_COUNTS.keys():
        harmful_list = harmful_words.get(category, [])
        benign_list = benign_words.get(category, [])
        
        category_mappings = {}
        min_count = min(len(harmful_list), len(benign_list))
        
        for i in range(min_count):
            category_mappings[harmful_list[i]] = benign_list[i]
        
        all_mappings[category] = category_mappings
    
    return all_mappings


def _validate_and_save_dictionary(
    dictionary: Dictionary,
    target_category: str,
    output_dir: str,
    task_num: Optional[int]
) -> None:
    """Validate dictionary and save to CSV file."""
    dictionary.validate()
    
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"task{task_num}_{target_category.lower()}.csv" if task_num else f"{target_category.lower()}.csv"
    file_path = os.path.join(output_dir, filename)
    
    dictionary.save_to_csv(file_path)


def _save_debug_response(
    response: str,
    context: str,
    category: str,
    count_needed: int,
    existing_count: int,
    task_num: Optional[int],
    output_dir: str,
    list_type: str
) -> None:
    """Save debug response to file."""
    debug_dir = os.path.join(output_dir, "..", "debug_responses")
    os.makedirs(debug_dir, exist_ok=True)
    
    if task_num is not None:
        debug_file = os.path.join(debug_dir, f"task{task_num}_{list_type}_{category}_raw.txt")
    else:
        debug_file = os.path.join(debug_dir, f"{list_type}_{category}_raw.txt")
    
    with open(debug_file, 'w', encoding='utf-8') as f:
        f.write(f"=== {list_type.upper()} {category.upper()} GENERATION ===\n")
        f.write(f"Context: {context}\n")
        f.write(f"Needed: {count_needed}\n")
        f.write(f"Existing words: {existing_count}\n\n")
        f.write(response)
