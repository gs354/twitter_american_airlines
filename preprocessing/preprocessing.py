"""
Functions used to pre-process the Twitter posts.
Some of the below functions were created by a colleague during another project; others are new or are adaptations.
"""

import re
from urllib.parse import urlparse

from bs4 import BeautifulSoup
import emoji


def check_text(text: str | list[str]) -> list[str]:
    """Ensures text is a string or list of strings

    Args:
        text (str, list): strings to be checked

    Raises:
        ValueError: If text is not a string or list of strings

    Returns:
        list: strings in list
    """
    if isinstance(text, str):
        text = [text]
    elif not (isinstance(text, list) and all(isinstance(s, str) for s in text)):
        raise ValueError("text argument must be a string or a list of strings")
    return text


def remove_emoji(text: str | list[str], replace: bool = False) -> list[str]:
    """Removes emoji or replaces it with a text description of the emoji

    Args:
        text (str, list): list of strings
        replace (bool, optional): Whether to replace or remove completely.
                                  Defaults to True.

    Returns:
        list: processed strings
    """

    text = check_text(text)

    processed_text = []

    for t in text:
        if replace:
            processed_text.append(emoji.demojize(t, delimiters=(" ", " ")))
        else:
            processed_text.append(emoji.replace_emoji(t, replace=""))

    return processed_text


def remove_urls(text: str | list[str]) -> list[str]:
    """Removes URLs from strings.

    Args:
        text (str, list): The input strings.

    Returns:
        list: The list of processed strings with URLs removed or replaced.
    """

    text = check_text(text)

    processed_text = []

    for t in text:
        words = t.split()
        filtered_words = [word for word in words if not is_url(word)]
        result_text = " ".join(filtered_words)
        processed_text.append(result_text)

    return processed_text


def is_url(word):
    # Use urlparse to check if the word is a URL
    parsed_url = urlparse(word)
    return all([parsed_url.scheme, parsed_url.netloc])


def remove_html(text: str | list[str]) -> list[str]:
    """Removes html code from strings

    Args:
        text (str, list): strings

    Returns:
        list: processed strings
    """

    text = check_text(text)
    processed_text = []

    for t in text:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(t, "html.parser")
        # Extract the text without HTML tags
        clean_text = soup.get_text(separator=" ")
        processed_text.append(clean_text)

    return processed_text


def remove_symbols(
    text: str | list[str],
    symbols: str | list[str],
    remove_keyword: bool | list[bool],
) -> list[str]:
    """Removes symbols and (optionally) associated words.

    Args:
        text (str, list): strings
        symbols (str, list): Symbols to be removed.
        remove_keyword (bool, list): Specify if the text adjacent to each
                                          symbol is removed.

    Returns:
        list: processed strings
    """

    text = check_text(text)

    if isinstance(symbols, str):
        symbols = [symbols]
    elif not isinstance(symbols, list):
        raise ValueError("symbols must be a string or a list of strings")

    if isinstance(remove_keyword, (bool)):
        remove_keyword = [remove_keyword] * len(symbols)
    elif not (
        isinstance(remove_keyword, list)
        and all(isinstance(s, bool) for s in remove_keyword)
    ):
        raise ValueError("remove_keyword must be a bool or a list of bools")

    processed_text = []
    for t in text:
        for idx, s in enumerate(symbols):
            r = remove_keyword[idx]
            if isinstance(r, bool) and r:
                # Remove symbol and associated text
                t = re.sub(s + r"\w+\s*", "", t).strip()
            else:
                # Remove only the symbol
                t = re.sub(re.escape(s), "", t)
        processed_text.append(t)

    return processed_text


def replace_curly_quotes(text: str | list[str]) -> list[str]:
    """Replaces unicode curly quotes with ascii ones

    Args:
        text (str, list): list of strings

    Returns:
        list: processed strings
    """
    text = check_text(text)

    # Unicode code points for curly quotes
    curly_quotes = {
        "\u2018": "'",  # Left single quotation mark
        "\u2019": "'",  # Right single quotation mark
        "\u201C": '"',  # Left double quotation mark
        "\u201D": '"',  # Right double quotation mark
        "\u2033": '"',  # Double prime (often used as a double quote)
        "\u2036": '"',  # Reversed double prime (also used as a double quote)
    }

    # Process each text in the list
    processed_text = []
    for t in text:
        for curly, straight in curly_quotes.items():
            t = t.replace(curly, straight)
        processed_text.append(t)

    return processed_text


def remove_whitespace_currency(text: str | list[str]) -> list[str]:
    """Removes whitespaces between unicode currency

    Args:
        text (str, list): strings to be processed

    Returns:
        list: processed text
    """
    # This regex looks for the £, € or $ symbol followed by any number
    # of whitespace characters
    # (\s*) and then a number (\d)
    pattern = r"([£€$])\s*(\d)"

    text = check_text(text)

    processed_text = []

    for t in text:
        # The substitution will replace the found pattern with the
        # currency symbol immediately followed by the number with no space
        processed_text.append(re.sub(pattern, r"\1\2", t))

    return processed_text


def fix_whitespace(text: str | list[str]) -> list[str]:
    """Removes excess/unneeded whitespace including spaces
       before sentence-ending punctuation and around quotes.

    Args:
        text (str, list): string or list of strings to be processed

    Returns:
        list: processed strings
    """

    text = check_text(text)

    processed_text = []

    for t in text:
        # Remove excess whitespace
        t = re.sub(r"\s+", " ", t)
        # Remove space before punctuation
        t = re.sub(r"\s+([?.!,:])", r"\1", t)
        # Remove space after an opening parenthesis and before a closing
        # parenthesis
        t = re.sub(r"\(\s+", "(", t)
        t = re.sub(r"\s+\)", ")", t)
        # Remove space after an opening quote and before a closing quote
        t = re.sub(r"(\s|^)(\"|\')\s", r"\1\2", t)
        t = re.sub(r"\s(\"|\')(\s|$)", r"\1\2", t)
        # Add space after sentence-ending punctuation if it's not there
        t = re.sub(r"([?.!])([^\s])(?=[A-Za-z])", r"\1 \2", t)
        # Add space after a comma if it's not there, and it's not followed by
        # a digit
        t = re.sub(r",([^\s\d])", r", \1", t)
        t = t.strip()
        processed_text.append(t)

    return processed_text


def default_preprocessing():
    """Returns a list of default preprocessing steps

    Returns:
        list: list of dict which contain function names and optional attributes
    """
    preprocessing_steps = [
        {"name": "remove_emoji", "attributes": {"replace": True}},
        {"name": "remove_urls"},
        {"name": "remove_html"},
        {
            "name": "remove_symbols",
            "attributes": {"symbols": ["#", "@"], "remove_keyword": [True, True]},
        },
        {"name": "replace_curly_quotes"},
        {"name": "remove_whitespace_currency"},
        {"name": "fix_whitespace"},
    ]

    return preprocessing_steps


def clean_text(
    text: str | list[str],
    preprocessing_steps: (dict | list[dict]) = None,
    verbose: bool = True,
) -> list[str]:
    """Carries out sequential pre-processing on text

    Args:
        text (str, list): _description_
        preprocessing_steps (list, optional): list of dicts containing
                                              function names and optional
                                              attributes. If None, default
                                              steps are loaded.
        verbose (bool, optional): Set True to display what cleaning
                                  functions are being applied.
                                  Default is True.

    Raises:
        ValueError: preporocessing_steps must be a list or dict
        ValueError: all elements in preprocessing_steps must be dictionaries
    Returns:
        list: processed strings
    """

    if preprocessing_steps is None:
        preprocessing_steps = default_preprocessing()

    elif isinstance(preprocessing_steps, dict):
        preprocessing_steps = [preprocessing_steps]
    elif not (
        isinstance(preprocessing_steps, list)
        and all(isinstance(s, dict) for s in preprocessing_steps)
    ):
        raise ValueError(
            """preprocessing_steps must be dictionary or list of dictionaries if
                            specified"""
        )

    for s in preprocessing_steps:
        f_name = s["name"]
        f_args = s.get("attributes", {})

        func = globals().get(f_name)
        if not func:
            raise ValueError(f"No pre-processing function '{f_name}' found.")

        if verbose:
            if f_args:
                print(f"Calling {f_name} with attributes {f_args}")
            else:
                print(f"Calling {f_name}")

        text = func(text, **f_args)

    return text
