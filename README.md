Clustering analysis of twitter [data](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) related to six American airlines in February 2015.

The tweet text is preprocessed and then passed to a sentence-transformer model to create embeddings to encode semantic content.

[BERTopic](https://maartengr.github.io/BERTopic/index.html), incorporating [UMAP](https://umap-learn.readthedocs.io/en/latest/) dimensionality reduction and [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html) density-based clustering is then used to assign topic labels to semantically similar tweets.

Monitoring the prevalence of particular themes (e.g. complaints, delays, cancellations, satisfaction) by operator, or looking at operator-led engagement levels are potential use cases for operators themselves or regulatory bodies. This would work with on-going data acquisition through the Twitter API and time series analysis in the form of dashboards or other analytics. 
