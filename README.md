Clustering analysis of twitter [data](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) related to American airlines.

The tweet text is preprocessed and then passed to a sentence-transformer model to create embeddings to encode semantic content.

[BERTopic](https://maartengr.github.io/BERTopic/index.html), incorporating [UMAP](https://umap-learn.readthedocs.io/en/latest/) dimensionality reduction and [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html) density-based clustering is then used to assign topic labels to semantically similar tweets.

Monitoring the prevalence of particular themes by operator is a potential use case for operators themselves or, e.g. regulatory bodies.