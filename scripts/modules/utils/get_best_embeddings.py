#!/usr/bin/env python3
import os
import shutil
import sys
import pickle
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
try:
    import scripts  # noqa: F401
except Exception as e:
    print(f"Warning: could not import 'scripts' package needed for unpickling: {e}", file=sys.stderr)

src = os.path.abspath("results/embeddings/test/kuaisim")
dest = os.path.abspath("best_embeddings")  # always best_embeddings
overwrite = True  # always true

if not os.path.isdir(src):
    print(f"Source not a directory: {src}", file=sys.stderr)
    sys.exit(1)

os.makedirs(dest, exist_ok=True)

for name in sorted(os.listdir(src)):
    s = os.path.join(src, name)
    if not os.path.isdir(s):
        continue
    d = os.path.join(dest, name)
    if os.path.exists(d) and overwrite:
        shutil.rmtree(d)
    if os.path.exists(d):
        print(f"Skipping existing: {d}")
        continue
    shutil.copytree(s, d)
    print(f"Copied {s} -> {d}")

# 2. Build augmented CSVs using embeddings now present in best_embeddings
base_csv = os.path.join(dest, "video_features_basic_Pure_fillna.csv")
if not os.path.isfile(base_csv):
    print(f"Base CSV not found: {base_csv}")
    sys.exit(0)

try:
    df_base = pd.read_csv(base_csv)
except Exception as e:
    print(f"Failed to read base CSV: {e}", file=sys.stderr)
    sys.exit(1)

output_root = os.path.join(dest, "Output")
os.makedirs(output_root, exist_ok=True)

subfolders = [d for d in os.listdir(dest)
              if os.path.isdir(os.path.join(dest, d))
              and d not in ("Output",) and not d.startswith('.')]

if not subfolders:
    print("No embedding subfolders found to process.")
    sys.exit(0)

first_col = df_base.columns[0]

for name in sorted(subfolders):
    # Enter first inner subfolder inside each embedding folder (e.g., run directory)
    parent_dir = os.path.join(dest, name)
    inner_dirs = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
    if inner_dirs:
        emb_dir = os.path.join(parent_dir, sorted(inner_dirs)[0])
    else:
        print(f"Skipping '{name}' (no inner subfolder)")
        continue

    items_path = os.path.join(emb_dir, "items.npy")
    sparse_path = os.path.join(emb_dir, "sparse_repr.pkl")

    if not (os.path.exists(items_path) and os.path.exists(sparse_path)):
        print(f"Skipping '{name}' (missing items.npy or sparse_repr.pkl)")
        continue

    print(f"\nProcessing embeddings: {name}")

    try:
        with open(sparse_path, "rb") as f:
            sparse_repr = pickle.load(f)
        embeddings = np.load(items_path, allow_pickle=True)
    except Exception as e:
        print(f"Failed loading embeddings for '{name}': {e}")
        continue

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)

    item_id_to_embedding = {}
    for embedding_idx in range(len(embeddings)):
        try:
            item_id = sparse_repr.get_item_id([embedding_idx])[0]
        except Exception as e:
            print(f"Error getting item_id for embedding {embedding_idx} in '{name}': {e}")
            continue
        item_id_to_embedding[item_id] = embedding_idx

    feature_columns = [f"feature{i+1}" for i in range(embeddings.shape[1])]
    df = df_base.copy()
    for col in feature_columns:
        df[col] = np.nan

    for idx, item_id in enumerate(df[first_col].values):
        emb_idx = item_id_to_embedding.get(item_id)
        if emb_idx is not None:
            df.loc[idx, feature_columns] = embeddings[emb_idx]

    rows_before = len(df)
    df = df.dropna(subset=feature_columns).reset_index(drop=True)
    print(f"Dropped {rows_before - len(df)} rows with NaN embeddings. Final rows: {len(df)}")

    out_dir = os.path.join(output_root, name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "video_features_basic_Pure_fillna.csv")
    try:
        df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")
    except Exception as e:
        print(f"Failed to save CSV for '{name}': {e}")