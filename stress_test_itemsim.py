"""
Stress test for FAISS-based exact cosine similarity search.

This script generates a large dummy embedding matrix and times the execution
of FAISS IndexFlatIP top-k search on CPU or GPU.

Unlike the old stress test, this does NOT materialize the full n x n similarity
matrix, because FAISS computes top-k neighbors directly.

Usage:
  python stress_test_itemsim_faiss.py
"""

import time
import numpy as np
import torch
import sys
import os
import faiss

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def get_topk_cosine_faiss(embeddings, k):
    """
    Exact cosine similarity top-k using FAISS Flat index.
    """
    x = np.ascontiguousarray(embeddings.astype(np.float32))
    n, d = x.shape

    index_cpu = faiss.IndexFlatIP(d)

    if torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
    else:
        index = index_cpu

    index.add(x)

    # Search k+1 to remove self-match
    scores, neighbors = index.search(x, k + 1)

    return neighbors[:, 1:], scores[:, 1:]


def run_stress_test():
    # --- Test Parameters ---
    NUM_ITEMS = 800000
    EMBEDDING_DIM = 128

    # k values to test (instead of batch sizes)
    K_VALUES_TO_TEST = [10, 50, 100, 200]

    print("--- FAISS Item Similarity Stress Test ---")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"Device: {torch.cuda.get_device_name(0)}")
    else:
        print("Device: CPU")

    print(f"Embeddings shape: ({NUM_ITEMS}, {EMBEDDING_DIM})")
    print("-" * 40)

    # Create normalized dummy embeddings (cosine similarity!)
    dummy_embeddings = np.random.rand(NUM_ITEMS, EMBEDDING_DIM).astype(np.float32)
    dummy_embeddings /= np.linalg.norm(dummy_embeddings, axis=1, keepdims=True) + 1e-9

    results = {}

    for k in K_VALUES_TO_TEST:
        print(f"Testing FAISS with k = {k}...")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            start_mem = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()

        start_time = time.monotonic()

        try:
            neighbors, scores = get_topk_cosine_faiss(dummy_embeddings, k)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end_time = time.monotonic()
            duration = end_time - start_time

            # Sanity checks
            assert neighbors.shape == (NUM_ITEMS, k)
            assert scores.shape == (NUM_ITEMS, k)

            if torch.cuda.is_available():
                peak_mem = torch.cuda.max_memory_allocated()
                mem_used_gb = peak_mem / (1024**3)
                results[k] = f"{duration:.4f}s | Peak GPU mem: {mem_used_gb:.3f} GB"
            else:
                results[k] = f"{duration:.4f}s"

            print(f"  -> Success! {results[k]}")

        except Exception as e:
            results[k] = f"FAILED: {type(e).__name__}"
            print(f"  -> FAILED with error: {e}")

    print("\n--- Summary ---")
    for k, result in results.items():
        print(f"k = {k:<4} | Result: {result}")

    print("-" * 40)
    print("Test complete.")
    print("This measures exact cosine top-k search using FAISS Flat.")
    print("No n×n similarity matrix is ever allocated.")


if __name__ == "__main__":
    run_stress_test()
