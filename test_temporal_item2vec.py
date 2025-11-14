#!/usr/bin/env python
"""
Temporal Item2Vec integrity tests.
Run: python test_temporal_item2vec.py
Validates timestamp processing for:
 - Item2vec_temp_model (discrete split)
 - Item2vec_temp_aug_model (discrete augmented, time groups)
 - Item2vec_Temp_Cont_model (continuous weights)
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Make sure project root is on path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import scripts as kw  # project constants

# Import models
from scripts.recommenders.Item2vec.Item2vec_disc import Item2vec_temp_model
from scripts.recommenders.Item2vec.Item2Vec_disc_aug import Item2vec_temp_aug_model
from scripts.recommenders.Item2vec.Item2vec_cont import Item2vec_Temp_Cont_model

# Convenience column names (fallback if missing)
COL_USER = getattr(kw, 'COLUMN_USER_ID', 'user_id')
COL_ITEM = getattr(kw, 'COLUMN_ITEM_ID', 'item_id')
COL_TS = getattr(kw, 'COLUMN_TIMESTAMP', 'timestamp')
COL_DT = getattr(kw, 'COLUMN_DATETIME', 'datetime')
COL_DIFF = getattr(kw, 'COLUMN_TIME_DIFF', 'time_diff')
COL_CUM = getattr(kw, 'COLUMN_TIME_CUMSUM', 'time_cumsum')
COL_CUM_NORM = getattr(kw, 'COLUMN_TIME_CUMSUM_NORM', 'time_cumsum_norm')
COL_MEAN = getattr(kw, 'COLUMN_MEAN', 'time_mean')
COL_STD = getattr(kw, 'COLUMN_STD', 'time_std')


def _print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _print_user_diffs(df: pd.DataFrame, title: str):
    print(f"\n{title}")
    tmp = df.copy()
    tmp[COL_DT] = pd.to_datetime(tmp[COL_TS], unit='s')
    tmp = tmp.sort_values([COL_USER, COL_DT])
    tmp['raw_diff'] = tmp.groupby(COL_USER)[COL_DT].diff().dt.total_seconds().fillna(0).astype(int)
    for uid, grp in tmp.groupby(COL_USER):
        print(f"- User {uid}: timestamps={grp[COL_TS].tolist()} diffs={grp['raw_diff'].tolist()}")


def build_dataframe():
    # Two users with a distinct large gap for user A
    data = [
        # user A
        ('A', 'i1', 0),
        ('A', 'i2', 30),
        ('A', 'i3', 60),
        ('A', 'i4', 1200),  # large gap -> should trigger split / new time group
        ('A', 'i5', 1260),
        # user B (smaller early gap, then a larger one)
        ('B', 'i6', 10),
        ('B', 'i7', 20),
        ('B', 'i8', 90),
    ]
    df = pd.DataFrame(data, columns=[COL_USER, COL_ITEM, COL_TS])
    _print_header("1) Building synthetic dataframe")
    print(df)
    _print_user_diffs(df, "Raw per-user time diffs before any model processing:")
    return df


def _explain_quantiles(diffs: pd.Series, min_time_diff: int, time_exp: float):
    valid = diffs[diffs > min_time_diff]
    if len(valid) == 0:
        print(f"  - No diffs > {min_time_diff}; model uses np.inf guard.")
        return None, None, None
    q1 = valid.quantile(0.25)
    q3 = valid.quantile(0.75)
    thresh = q3 + time_exp * (q3 - q1)
    print(f"  - Using diffs>{min_time_diff}: {valid.tolist()} -> q1={q1:.2f}, q3={q3:.2f}, threshold={thresh:.2f}")
    return q1, q3, thresh


def test_disc_model():
    df = build_dataframe()
    model = Item2vec_temp_model(
        embedding_dir='/tmp/test_disc',
        w_size=1,
        time_exp=1.5,
        min_time_diff=20,
        epochs=1,
        factors=10,
    )
    _print_header("2) Discrete (split) model: computing timestamp_diff")

    # Explain expected quantiles per user before calling the model
    df_explain = df.copy()
    df_explain[COL_DT] = pd.to_datetime(df_explain[COL_TS], unit='s')
    df_explain = df_explain.sort_values([COL_USER, COL_DT])
    df_explain[COL_DIFF] = df_explain.groupby(COL_USER)[COL_DT].diff().dt.total_seconds().fillna(0)
    print("Expected thresholds per user (pre-computation):")
    for uid, grp in df_explain.groupby(COL_USER):
        print(f"- User {uid}:")
        _explain_quantiles(grp[COL_DIFF], model.min_time_diff, model.time_exp)

    processed = model.timestamp_diff(df.copy())

    print("\nProcessed dataframe (discrete split):")
    print(processed[[COL_USER, 'old_user_id', COL_ITEM, COL_TS, COL_DT, COL_DIFF, 'increment']])

    assert 'old_user_id' in processed.columns, 'old_user_id column missing in disc model output'
    assert 'increment' in processed.columns, 'increment column missing in disc model output'

    # For user A expect 2 segments (increment 0 then 1)
    user_a = processed[processed['old_user_id'] == 'A']
    increments = user_a['increment'].tolist()
    print(f"User A increments: {increments}")
    assert increments == [0, 0, 0, 1, 1], f'Disc increments incorrect: {increments}'

    # User IDs should have been remapped (different integer ids for segments)
    new_ids = user_a[COL_USER].unique()
    print(f"User A new remapped user ids (post-split): {new_ids}")
    assert len(new_ids) == 2, f'User A should be split into 2 new ids, found {len(new_ids)}'

    print('Disc model timestamp_diff passed.')


def test_disc_aug_model():
    df = build_dataframe()
    model = Item2vec_temp_aug_model(
        embedding_dir='/tmp/test_disc_aug',
        w_size=1,
        time_exp=1.5,
        min_time_diff=20,
        epochs=1,
        factors=10,
    )
    _print_header("3) Discrete (augmented) model: computing timestamp_diff")

    # Explain expected quantiles per user before calling the model
    df_explain = df.copy()
    df_explain[COL_DT] = pd.to_datetime(df_explain[COL_TS], unit='s')
    df_explain = df_explain.sort_values([COL_USER, COL_DT])
    df_explain[COL_DIFF] = df_explain.groupby(COL_USER)[COL_DT].diff().dt.total_seconds().fillna(0)
    print("Expected thresholds per user (pre-computation):")
    for uid, grp in df_explain.groupby(COL_USER):
        print(f"- User {uid}:")
        _explain_quantiles(grp[COL_DIFF], model.min_time_diff, model.time_exp)

    processed = model.timestamp_diff(df.copy())

    print("\nProcessed dataframe (discrete augmented):")
    cols_to_show = [COL_USER, COL_ITEM, COL_TS, COL_DT, COL_DIFF, 'increment']
    cols_to_show = [c for c in cols_to_show if c in processed.columns]
    print(processed[cols_to_show])

    assert 'increment' in processed.columns, 'increment column missing in disc_aug output'
    # Ensure user ids are NOT remapped (should remain A/B)
    assert 'old_user_id' not in processed.columns, 'disc_aug should not create old_user_id'

    user_a = processed[processed[COL_USER] == 'A']
    increments = user_a['increment'].tolist()
    print(f"User A increments (aug): {increments}")
    assert increments == [0, 0, 0, 1, 1], f'Disc_aug increments incorrect: {increments}'

    # For user B with only the last gap exceeding threshold, expect [0, 0, 1]
    user_b = processed[processed[COL_USER] == 'B']
    inc_b = user_b['increment'].tolist()
    print(f"User B increments (aug): {inc_b}")
    assert inc_b == [0, 0, 1], f'Disc_aug user B increments incorrect: {inc_b}'

    print('Disc_aug model timestamp_diff passed.')


def test_cont_model():
    df = build_dataframe()
    model = Item2vec_Temp_Cont_model(
        embedding_dir='/tmp/test_cont',
        w_size=1,
        min_time_diff=10,
        epochs=1,
        factors=10,
    )
    _print_header("4) Continuous model: computing timestamp_cum")

    processed = model.timestamp_cum(df.copy())

    print("\nProcessed dataframe (continuous weighting):")
    cols_to_show = [COL_USER, COL_ITEM, COL_TS, COL_DT, COL_DIFF, COL_CUM, COL_CUM_NORM, COL_MEAN, COL_STD]
    cols_to_show = [c for c in cols_to_show if c in processed.columns]
    print(processed[cols_to_show])

    required_cols = [COL_DIFF, COL_CUM, COL_CUM_NORM, COL_MEAN, COL_STD]
    for c in required_cols:
        assert c in processed.columns, f'Missing column {c} in cont model output'

    # Check cumsum monotonic per user
    for uid, grp in processed.groupby(COL_USER):
        diffs = np.diff(grp[COL_CUM])
        print(f"User {uid} cumulative steps: {grp[COL_CUM].tolist()} diffs={diffs.tolist() if len(diffs)>0 else []}")
        assert np.all(diffs >= 0), f'Cumulative time not monotonic for user {uid}'

    # Normalized weights within range
    wmin, wmax = processed[COL_CUM_NORM].min(), processed[COL_CUM_NORM].max()
    print(f"Normalized cumulative weights range: [{wmin:.3f}, {wmax:.3f}] (min_weight={model.min_weight})")
    assert processed[COL_CUM_NORM].between(model.min_weight, 1).all(), 'Normalized cumulative weights out of range'

    print('Cont model timestamp_cum passed.')


def run_all():
    _print_header('Running temporal Item2Vec tests...')
    test_disc_model()
    test_disc_aug_model()
    test_cont_model()
    _print_header('All temporal Item2Vec tests passed.')


if __name__ == '__main__':
    run_all()
