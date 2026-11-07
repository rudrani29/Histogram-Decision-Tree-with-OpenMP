"""Reference: scikit-learn exact regression tree on the same binary file and the same 80/20 split.
Usage:  pip install scikit-learn numpy ;  python bench/ref_sklearn.py data/d1m.bin [depth] [min_leaf]
File format (written by gen_data): int32 N, int32 D, float32 X[N*D] (row-major), float32 y[N].
"""
import csv
import os
import sys
import time
import numpy as np
from sklearn.tree import DecisionTreeRegressor


def load(path):
    with open(path, "rb") as f:
        n, d = np.frombuffer(f.read(8), dtype=np.int32)
        x = np.frombuffer(f.read(4 * int(n) * int(d)), dtype=np.float32).reshape(int(n), int(d))
        y = np.frombuffer(f.read(4 * int(n)), dtype=np.float32)
    return x, y


def rmse(p, y):
    return float(np.sqrt(np.mean((p.astype(np.float64) - y.astype(np.float64)) ** 2)))


if __name__ == "__main__":
    path = sys.argv[1]
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    min_leaf = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    X, y = load(path)
    n = X.shape[0]
    ntr = int(0.8 * n)  # same as the C++ code: train on [0, 0.8N), test on the rest
    m = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=min_leaf, random_state=0)
    t0 = time.perf_counter()
    m.fit(X[:ntr], y[:ntr])
    t = time.perf_counter() - t0
    rtr, rte = rmse(m.predict(X[:ntr]), y[:ntr]), rmse(m.predict(X[ntr:]), y[ntr:])
    nodes = m.tree_.node_count
    print(f"sklearn N={n} depth={depth} fit_time={t:.3f}s rmse_train={rtr:.5f} rmse_test={rte:.5f} nodes={nodes}")
    out = "bench/final/sklearn.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fresh = not os.path.exists(out)
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if fresh:
            w.writerow(["N", "depth", "min_leaf", "fit_time", "rmse_train", "rmse_test", "nodes"])
        w.writerow([n, depth, min_leaf, f"{t:.4f}", f"{rtr:.5f}", f"{rte:.5f}", nodes])