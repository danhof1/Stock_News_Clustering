# The Good, The Bad, and The Cluster

This repository contains a Jupyter Notebook (`The_good,_the_bad,_and_the_cluster_Master_file.ipynb`) that focuses on clustering news headlines. The notebook demonstrates data preprocessing, news category classification, text embedding, dimensionality reduction, and clustering techniques applied to a dataset of news headlines.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Models Used](#models-used)
- [Dependencies](#dependencies)

## Overview

The main goal of this project is to categorize and cluster news headlines. It processes news data, classifies headlines into predefined categories, and then applies clustering algorithms to group similar headlines together. This can be useful for understanding trends in news, organizing large datasets of articles, or content recommendation.

## Features

- **Data Loading and Preparation**: Downloads and extracts news headline datasets.
- **News Category Classification**: Utilizes a pre-trained Hugging Face transformer model (`ilsilfverskiold/classify-news-category-iptc`) to classify news headlines into various categories (e.g., economy, science, politics, sport).
- **Text Embedding**: Converts news titles into numerical embeddings using a financial domain-specific SentenceTransformer model (`FinLang/finance-embeddings-investopedia`).
- **Dimensionality Reduction**: Applies Truncated SVD (Singular Value Decomposition) to reduce the dimensionality of the text embeddings while preserving a significant portion of the original variance (80-90%).
- **Clustering**: Employs the Elbow Method for K-Means to determine an optimal number of clusters for the news headlines.

## Installation

To run this notebook, you'll need to have Python and Jupyter installed. It's recommended to use a virtual environment.

```bash
# Clone the repository (if applicable)
# git clone <repository-url>
# cd The_good,_the_bad,_and_the_cluster

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install the required packages
pip install pandas numpy scikit-learn matplotlib seaborn sentence-transformers nltk transformers

```

## Usage

1.  **Download Data**: The notebook automatically downloads the `news_headline_clustering.zip` file containing `news_curated.par`, `news_good.par`, and `news_bad.par` upon execution.
2.  **Run the Jupyter Notebook**: Open the `The_good,_the_bad,_and_the_cluster_Master_file.ipynb` file in Jupyter Lab or Jupyter Notebook and run all cells.

    ```bash
    jupyter notebook
    ```
    or
    ```bash
    jupyter lab
    ```

    The notebook will execute the following steps:
    - Load the datasets.
    - Classify news categories.
    - Generate sentence embeddings.
    - Perform dimensionality reduction.
    - Apply clustering algorithms.

## Data

The project uses news headline data from the following parquet files, which are extracted from `news_headline_clustering.zip`:
- `news_curated.par`
- `news_good.par`
- `news_bad.par`

These files primarily contain news titles used for classification and clustering.

## Models Used

-   **Text Classification**: `ilsilfverskiold/classify-news-category-iptc` (Hugging Face Transformers)
-   **Sentence Embedding**: `FinLang/finance-embeddings-investopedia` (Sentence Transformers)

## Dependencies

The key libraries required are:
-   `pandas`
-   `numpy`
-   `scikit-learn` (for TF-IDF, KMeans, MiniBatchKMeans, metrics, TruncatedSVD, Normalizer, StandardScaler)
-   `matplotlib`
-   `seaborn`
-   `sentence_transformers`
-   `nltk`
-   `transformers`

## Results

This project yielded several key results through its various stages of analysis and modeling:

* **Sentiment Analysis**: Initial exploration into sentiment analysis using `mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis` indicated that most news headlines were classified as neutral, rendering sentiment analysis less useful for differentiating clusters within this specific dataset.

* **News Category Classification**: The application of the `ilsilfverskiold/classify-news-category-iptc` model successfully classified news headlines into 15 distinct categories. The distribution of these categories revealed the predominant themes within the dataset, with "economy, business, and finance" being the most frequent category.

    | News Category                              | Count |
    | :----------------------------------------- | :---- |
    | economy, business, and finance             | 1554  |
    | science and technology                     | 487   |
    | politics                                   | 380   |
    | sport                                      | 138   |
    | disaster, accident, and emergency incident | 132   |
    | conflict, war and peace                    | 124   |
    | arts, culture, entertainment and media     | 100   |
    | environment                                | 77    |
    | human interest                             | 74    |
    | crime, law and justice                     | 73    |
    | health                                     | 47    |
    | society                                    | 18    |
    | lifestyle and leisure                      | 12    |
    | weather                                    | 9     |
    | labour                                     | 6     |

* **Dimensionality Reduction**: Using Truncated SVD, the dimensionality of the sentence embeddings was successfully reduced while retaining a high percentage of the original variance. Specifically, the model achieved an "Explained variance: 0.91" (91% of the variance) with 200 components, demonstrating effective feature compression for subsequent clustering.

* **Clustering (Elbow Method)**: The notebook employed the Elbow Method to identify an optimal number of clusters for the news headlines based on their reduced-dimension embeddings, facilitating meaningful grouping of similar news content.
