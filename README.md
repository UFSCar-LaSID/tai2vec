
# Time embeddings

## Installation

Before executing the scripts, it is important to install the necessary datasets and Python libraries. The following two subsections explain how to do that.

### Python libraries

The recommended way to install the Python libraries necessary to run the experiments is to use an Anaconda environment. You can create it with the command below:

```
conda env create --name tai2vec --file=environment.yml
```

With the environment created, you must activate it to be able to run the experiments:

```
conda activate tai2vec
```

Another way to install the libraries is using the `requirements.txt` file (although this is not recommended). The Python version 3.10.18 is recommended. You can then install the libraries with the command below:

```
python -m pip install -r requirements.txt
```

### Datasets

Downloading the datasets is necessary to run the experiments. A list with download link and where to save the files are given below:

- [AmazonBeauty](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/All_Beauty.jsonl.gz): put `All_Beauty.jsonl` file in `raw/amazon-beauty`
- [AmazonBooks](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews): download and extract in `raw/amazon-books`
- [BestBuy](https://www.kaggle.com/c/acm-sf-chapter-hackathon-big/data?select=train.csv): put `train.csv` file in `raw/bestbuy`
- [CiaoDVD](https://guoguibing.github.io/librec/datasets.html): download `ciaodvd.zip` and extract `movie-ratings.txt` file in `raw/ciaodvd`
- [MovieLens-100K](https://grouplens.org/datasets/movielens/): download `ml-100k.zip` in the `MovieLens 100K Dataset` section and extract it in `raw/ml-100k`
- [MovieLens-1M](https://grouplens.org/datasets/movielens/): download `ml-1m.zip` in the `MovieLens 1M Dataset` section and extract it in `raw/ml-1m`

## Executing the experiments

With all the installation done, you can proceed to the scripts execution. The following subsections explain the necessary scripts to execute in order to reproduce our results.

### Preprocess datasets

With the raw datasets downloaded, it's necessary to preprocess them before generating the recommendations.
To do that, execute the following command:

```
python src/scripts/preprocess.py
```

Executing this Python code will ask you which datasets to preprocess. Input the dataset indexes separated by a space to select the datasets.

Another way to select the datasets is by executing the command below:

```
python scripts/preprocess.py --datasets <datasets>
```

Replace `<datasets>` with the names (or indexes) of the datasets separated by commas (","). The available datasets to preprocess are:

- \[1\]: amazon-beauty
- \[2\]: amazon-books
- \[3\]: bestbuy
- \[4\]: ciaodvd
- \[5\]: ml-100k
- \[6\]: ml-1m
- all (it will use all datasets)

### Run the recommendation methods

After preprocessing the datasets, you can proceed to the recommendation and evaluation script (main). You can execute it with:

```
python scripts/main.py
```

Executing this Python code will ask you which datasets and recommenders to use. Input the datasets and recommenders indexes separated by spaces to select them.

Another way to select the datasets and recommenders is by executing the command below:

```
python scripts/main.py --datasets <datasets> --recommenders <recommenders>
```

Replace `<datasets>` with the names (or indexes) of the datasets separated by commas (","). The available datasets to use are:

- \[1\]: amazon-beauty
- \[2\]: amazon-books
- \[3\]: bestbuy
- \[4\]: ciaodvd
- \[5\]: ml-100k
- \[6\]: ml-1m
- all (it will use all datasets)

Replace `<recommenders>` with the names (or indexes) of the recommenders separated by commas (","). The available recommenders to use are:

- \[1\]: ALS
- \[2\]: BPR
- \[3\]: Item2Vec
- \[4\]: TimeI2V_Disc_Aug
- \[5\]: TimeI2V_Cont
- all (it will use all recommenders)

### Generate metrics and plots

After executing the main code, you can plot the results. To do so, execute the following command:

```
python scripts/generate_plots.py
```

Executing this Python code will ask you which datasets and recommenders to use. Input the datasets and recommenders indexes separated by spaces to select them.

Another way to select the datasets and recommenders is by executing the command below:

```
python scripts/generate_plots.py --datasets <datasets> --recommenders <recommenders>
```

Replace `<datasets>` with the names (or indexes) of the datasets separated by commas (","). The available datasets to use are:

- \[1\]: amazon-beauty
- \[2\]: amazon-books
- \[3\]: bestbuy
- \[4\]: ciaodvd
- \[5\]: ml-100k
- \[6\]: ml-1m
- all (it will use all datasets)

Replace `<recommenders>` with the names (or indexes) of the recommenders separated by commas (","). The available recommenders to use are:

- \[1\]: ALS
- \[2\]: BPR
- \[3\]: Item2Vec
- \[4\]: TimeI2V_Disc_Aug
- \[5\]: TimeI2V_Cont
- all (it will use all recommenders)