
# Time embeddings

## Installation

### Python libraries

The recommended way to install the python libaries necessary to run the experiments is using an Anaconda environment. You can create it with the command below:

```
conda env create --name tai2vec --file=environment.yml
```

With the envinroment created, you must activate it to be able to run the experiments:

```
conda activate tai2vec
```

### Datasets

Downloading the datasets is necessary to run the experiments. A list with download link and where to save the files are given below:

- [AmazonBeauty](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/All_Beauty.jsonl.gz): put `All_Beauty.jsonl` file in `raw/amazon-beauty`
- [AmazonBooks](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews): download and extract in `raw/amazon-books`
- [BestBuy](https://www.kaggle.com/c/acm-sf-chapter-hackathon-big/data?select=train.csv): put `train.csv` file in `raw/BestBuy`
- [MovieLens-100K](https://grouplens.org/datasets/movielens/): download `ml-100k.zip` in the `MovieLens 100K Dataset` section and extract it in `raw/ml-100k`
- [MovieLens-1M](https://grouplens.org/datasets/movielens/): download `ml-1m.zip` in the `MovieLens 1M Dataset` section and extract it in `raw/ml-1m`

## Executing the experiments

### Preprocess datasets

### Run the recommendation methods

### Generate metrics and plots
