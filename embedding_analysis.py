from __future__ import annotations

import os
import pickle
from typing import Dict

import numpy as np
import pandas as pd

import scripts as kw
from scripts.recommenders.itemSim import combine_embeddings, get_cosine_similarity_matrix


def _first_run_dir(base_results_dir: str, recommender: str) -> str:
    base_dir = os.path.join(base_results_dir, recommender)
    run_folders = sorted(
        d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
    )
    if not run_folders:
        raise FileNotFoundError(f"No run folders in: {base_dir}")
    return os.path.join(base_dir, run_folders[0])


def _load_embeddings(run_dir: str) -> np.ndarray:
    target = np.load(os.path.join(run_dir, kw.FILE_ITEMS_EMBEDDINGS))
    context = np.load(os.path.join(run_dir, kw.FILE_CONTEXT_EMBEDDINGS))
    return combine_embeddings(
        target,
        context,
        combination_strategy="avg_norm_after",
        use_norm=True,
    )


DATASET_NAME = "ml-1m"
RECOMMENDERS = ["Item2Vec_itemSim", "TimeI2V_Disc_Aug", "TimeI2V_Cont"]

MOVIE_ID = 1974
TOP_N = 20

movies_path = os.path.join("datasets", DATASET_NAME, "movies.csv")
movies_df = pd.read_csv(movies_path, sep=";", quotechar='"')

required_cols = {"id_item", "title"}

movies_df = movies_df.rename(columns={"id_item": "movieId"})

base_results_dir = os.path.join("results", "embeddings", "test", DATASET_NAME)

run_dirs = {rec: _first_run_dir(base_results_dir, rec) for rec in RECOMMENDERS}

with open(os.path.join(run_dirs[RECOMMENDERS[0]], kw.FILE_SPARSE_REPR), "rb") as f:
    data_repr = pickle.load(f)

movie_inner = int(data_repr.get_item_index(int(MOVIE_ID)))

sims: Dict[str, np.ndarray] = {}

for rec, run_dir in run_dirs.items():
    emb = _load_embeddings(run_dir)
    sims[rec] = get_cosine_similarity_matrix(emb, use_norm=False)

query_title = movies_df.loc[movies_df["movieId"] == int(MOVIE_ID), "title"]
query_title = query_title.iloc[0] if not query_title.empty else str(MOVIE_ID)
print(f"Query: {MOVIE_ID} - {query_title}\n")

blocks = []
for rec, sim in sims.items():
    scores = sim[movie_inner].copy()
    scores[movie_inner] = -np.inf

    k = min(int(TOP_N), scores.shape[0] - 1)
    top_idx = np.argpartition(-scores, k)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    neighbor_ids = data_repr.get_item_id(top_idx)
    df = pd.DataFrame({"movieId": neighbor_ids})
    df = df.merge(movies_df[["movieId", "title"]], on="movieId", how="left")
    df["title"] = df["title"].fillna(df["movieId"].astype(str))

    df = df[["title"]].reset_index(drop=True)
    df = df.rename(columns={"title": f"{rec} | title"})
    blocks.append(df)

out = pd.concat(blocks, axis=1)

with pd.option_context("display.max_rows",None,"display.max_columns",None,"display.width",220,"display.max_colwidth",80,):
    print(out)