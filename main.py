"""
Main entry point for CAT attack pipeline.
"""
from src.utils import run_attack_pipeline

# Default categories
CATEGORIES = ["Education", "Entertainment", "Health", "Business"]

# Default model
DEFAULT_MODEL = "gpt-4o-mini"


def main():
    """Main entry point."""
    run_attack_pipeline(
        categories=CATEGORIES,
        model_name=DEFAULT_MODEL,
        limit=2
    )


if __name__ == "__main__":
    main()
