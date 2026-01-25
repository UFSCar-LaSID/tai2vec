
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

Another way to install the libraries is with the `requirements.txt` (but not so recommended). The Python version 3.10.18 is recommended. You can then install the libraries with the command below:

```
python -m pip install -r requirements.txt
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

With the raw datasets downloaded, it's necessary to preprocess them before generating the recommendations.
To do that, execute the following command:

```
python src/scripts/preprocess.py
```

Executing this Python code will ask you which datasets to preprocess. Input the datasets indexes separated by space to select the datasets.

Another way to select the datasets is by executing the command below:

```
python scripts/preprocess.py --datasets <datasets>
```

Replace `<datasets>` with the names (or indexes) of the datasets separated by comma (","). The available datasets to preprocess are:

- \[1\]: amazon-beauty
- \[2\]: amazon-books
- \[3\]: bestbuy
- \[4\]: ciaodvd
- \[5\]: ml-100k
- \[6\]: ml-1m
- all (it will use all datasets)

### Run the recommendation methods

### Generate metrics and plots
