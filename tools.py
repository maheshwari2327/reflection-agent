from langchain_core.tools import tool


@tool
def count_words(text: str) -> str:
    """Count the words in a piece of text and report whether it is short, medium or long."""
    count = len(text.strip().split()) if text.strip() else 0
    if count < 50:
        category = "short"
    elif count <= 90:
        category = "medium"
    else:
        category = "long"
    return f"{count} words ({category})"
