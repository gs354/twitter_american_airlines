"""
Functions used to pre-process the Twitter posts.
"""

import re
from urllib.parse import urlparse
import pandas as pd
from bs4 import BeautifulSoup
import nltk
import emoji
from spellchecker import SpellChecker
from nltk.tokenize import TweetTokenizer


def replace_substring(
    df: pd.DataFrame, col: str, str_to_replace: str, replacement: str
) -> pd.DataFrame:
    """Replaces a substring with specified replacement word.
    Capitalisation of first letter of replacement is applied
    if the str_to_replace starts the string or comes after
    sentence-ending punctuation.

    Args:
        df: pandas dataframe
        col: column name on which to operate
        str_to_replace: substring to be replaced
        replacement: replacement string

    Returns:
        pandas dataframe
    """

    df[col] = df[col].apply(
        lambda x: x.replace(str_to_replace, replacement)
        if not x.startswith(str_to_replace)
        and not is_substring_after_punctuation(x, str_to_replace)
        else x.replace(str_to_replace, replacement.title())
    )
    return df


def is_substring_after_punctuation(sentence: str, substring: str) -> bool:
    """Checks if substring comes after sentence-ending punctuation.

    Args:
        sentence (str): text to examine
        substring (str): substring to test

    Returns:
        bool
    """
    pattern = re.compile(rf"[.!?]\s*{re.escape(substring)}")
    return bool(pattern.search(sentence))


def find_incorrect_spellings(
    df: pd.DataFrame, col: str, distance: int = 2
) -> pd.DataFrame:
    """Checks text against words in the SpellChecker corpus
    and returns a column containing dictionaries where the
    keys are the potentially misspelled and the values are
    the suggested correction.

    Args:
        df: pandas dataframe
        col: column name on which to operate
        distance: the maximum edit distance (how many alterations
        permitted for a correction from the original word)

    Returns:
        pandas dataframe
    """
    # Initialize TweetTokenizer and SpellChecker
    tokenizer = TweetTokenizer()
    spell_checker = SpellChecker(distance=distance)
    ordinal_number_pattern = re.compile(r"^\d+(st|nd|rd|th)$", re.IGNORECASE)

    processed_text = []
    # Cache suggestions to avoid re-computing corrections
    suggestions_dict = {}
    # Process each string in the series
    for t in df[col]:
        # Tokenize the string using TweetTokenizer
        tokens = tokenizer.tokenize(t)
        # Reduce to list of unknown words
        tokens = spell_checker.unknown(tokens)
        # Store the unusual spellings and potential corrections in a dict
        unusual_spellings = {}
        for token in tokens:
            if (
                token.startswith("@")
                or token.startswith("#")
                or not token.isascii()
                or ordinal_number_pattern.match(token)
            ):
                continue
            elif token in suggestions_dict:
                unusual_spellings[token] = suggestions_dict[token]
            else:
                spell_checked = spell_checker.correction(token)
                if token != spell_checked:
                    unusual_spellings[token] = spell_checked
                    suggestions_dict[token] = spell_checked

        processed_text.append(unusual_spellings)

    df["check_spellings"] = processed_text
    return df


def unusual_words(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Checks text against words in the nltk english words corpus
    and returns a column with the unusual words.

    Args:
        df: pandas dataframe
        col: column name on which to operate

    Returns:
        pandas dataframe
    """
    nltk.download("words")
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())
    processed_text = []
    for text in df[col]:
        text_vocab = set(w.lower() for w in text.split() if w.isalpha())
        unusual = text_vocab - english_vocab
        processed_text.append(sorted(unusual))
    df["unusual_words"] = processed_text
    return df


def remove_emoji(df: pd.DataFrame, col: str, replace: bool = False) -> pd.DataFrame:
    """Removes emoji or replaces it with a text description of the emoji

    Args:
        df: pandas dataframe
        col: column name on which to operate
        replace (bool, optional): Whether to replace or remove completely.

    Returns:
        pandas dataframe
    """

    processed_text = []

    for t in df[col]:
        if replace:
            processed_text.append(emoji.demojize(t, delimiters=(" ", " ")))
        else:
            processed_text.append(emoji.replace_emoji(t, replace=""))

    df[col] = processed_text
    return df


def remove_urls(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Removes URLs from strings.

    Args:
        df: pandas dataframe
        col: column name on which to operate

    Returns:
        pandas dataframe
    """

    processed_text = []

    for t in df[col]:
        words = t.split()
        filtered_words = [word for word in words if not is_url(word)]
        result_text = " ".join(filtered_words)
        processed_text.append(result_text)

    df[col] = processed_text
    return df


def is_url(word):
    # Use urlparse to check if the word is a URL
    parsed_url = urlparse(word)
    return all([parsed_url.scheme, parsed_url.netloc])


def remove_html(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Removes html code from strings

    Args:
        df: pandas dataframe
        col: column name on which to operate

    Returns:
        pandas dataframe
    """

    processed_text = []

    for t in df[col]:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(t, "html.parser")
        # Extract the text without HTML tags
        clean_text = soup.get_text(separator=" ")
        processed_text.append(clean_text)

    df[col] = processed_text
    return df


def remove_symbols(
    df: pd.DataFrame,
    col: str,
    symbols: str | list[str],
    remove_keyword: bool | list[bool],
) -> pd.DataFrame:
    """Removes symbols and (optionally) associated words.

    Args:
        df: pandas dataframe
        col: column name on which to operate
        symbols (str, list): Symbols to be removed.
        remove_keyword (bool, list): Specify if the text adjacent to each
                                          symbol is removed.

    Returns:
        pandas dataframe
    """

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
    for t in df[col]:
        for idx, s in enumerate(symbols):
            r = remove_keyword[idx]
            if isinstance(r, bool) and r:
                # Remove symbol and associated text
                t = re.sub(s + r"\w+\s*", "", t).strip()
            else:
                # Remove only the symbol
                t = re.sub(re.escape(s), "", t)
        processed_text.append(t)

    df[col] = processed_text
    return df


def replace_curly_quotes(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Replaces unicode curly quotes with ascii ones

    Args:
        df: pandas dataframe
        col: column name on which to operate

    Returns:
        pandas dataframe
    """

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
    for t in df[col]:
        for curly, straight in curly_quotes.items():
            t = t.replace(curly, straight)
        processed_text.append(t)

    df[col] = processed_text
    return df


def remove_whitespace_currency(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Removes whitespaces between unicode currency

    Args:
        df: pandas dataframe
        col: column name on which to operate

    Returns:
        pandas dataframe
    """
    # This regex looks for the £, € or $ symbol followed by any number
    # of whitespace characters
    # (\s*) and then a number (\d)
    pattern = r"([£€$])\s*(\d)"

    processed_text = []

    for t in df[col]:
        # The substitution will replace the found pattern with the
        # currency symbol immediately followed by the number with no space
        processed_text.append(re.sub(pattern, r"\1\2", t))

    df[col] = processed_text
    return df


def fix_whitespace(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Removes excess/unneeded whitespace including spaces
       before sentence-ending punctuation and around quotes.

    Args:
        df: pandas dataframe
        col: column name on which to operate

    Returns:
        pandas dataframe
    """

    processed_text = []

    for t in df[col]:
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

    df[col] = processed_text
    return df
