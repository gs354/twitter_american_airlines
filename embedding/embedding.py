"""
This script applies sentence_transformer models to text columns held in Pandas dataframes.
The output is saved in .npy format.
"""

from os.path import join
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

READ_DIR = "../data/processed"

SAVE_DIR = "../data/processed"

FILENAME = "twitter_airline_sentiment_cleaned_emoji_urls_html_symbols@#_quotes_currency_whitespace_3wordtweetdrop"

MODEL_NAMES = ["all-mpnet-base-v2"]

TEXT_COLUMN = "clean_text"

if __name__ == "__main__":
    # Read in FILENAME as csv
    df = pd.read_csv(f"{READ_DIR}/{FILENAME}.csv")

    for model in MODEL_NAMES:
        embedding_model = SentenceTransformer(model)
        encoded_text = embedding_model.encode(df[TEXT_COLUMN])

        savename = FILENAME + "_" + model
        np.save(join(SAVE_DIR, savename), encoded_text)
