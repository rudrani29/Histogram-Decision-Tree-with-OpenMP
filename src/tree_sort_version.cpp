#include "common.h"
#include <algorithm>
#include <cmath>
#include <omp.h>

namespace {
struct ENode { int feature = -1, left = -1, right = -1; float thr = 0; double value = 0; };

int grow_exact(const Dataset& ds, const Params& p, std::vector<ENode>& nodes,
               std::vector<int> rows, int depth) {
    int id = nodes.size(); nodes.emplace_back();
    size_t n = rows.size(); double S = 0;
    for (int r : rows) S += ds.y[r];
    nodes[id].value = n ? S / n : 0;
    if (depth >= p.max_depth || n < 2u * (size_t)p.min_leaf) return id;

    double base = S * S / n, bestGain = -1; int bf = -1; float bthr = 0;
    std::vector<std::pair<float, float>> v(n);
    for (int f = 0; f < ds.D; ++f) {
        for (size_t k = 0; k < n; ++k) v[k] = {ds.X_raw[(size_t)rows[k] * ds.D + f], ds.y[rows[k]]};
        std::sort(v.begin(), v.end());
        double SL = 0;
        for (size_t k = 0; k + 1 < n; ++k) {
            SL += v[k].second;
            size_t nL = k + 1, nR = n - nL;
            if (v[k].first == v[k + 1].first) continue;
            if (nL < (size_t)p.min_leaf || nR < (size_t)p.min_leaf) continue;
            double SR = S - SL;
            double gain = SL * SL / nL + SR * SR / nR - base;
            if (gain > bestGain) { bestGain = gain; bf = f; bthr = 0.5f * (v[k].first + v[k + 1].first); }
        }
    }
    if (bf < 0 || bestGain < p.min_gain) return id;
    std::vector<int> L, R;
    for (int i : rows) (ds.X_raw[(size_t)i * ds.D + bf] <= bthr ? L : R).push_back(i);
    nodes[id].feature = bf; nodes[id].thr = bthr;
    int l = grow_exact(ds, p, nodes, std::move(L), depth + 1);
    int r = grow_exact(ds, p, nodes, std::move(R), depth + 1);
    nodes[id].left = l; nodes[id].right = r;
    return id;
}

double predict_exact(const std::vector<ENode>& nodes, const Dataset& ds, int row) {
    int id = 0;
    while (nodes[id].left >= 0)
        id = ds.X_raw[(size_t)row * ds.D + nodes[id].feature] <= nodes[id].thr ? nodes[id].left : nodes[id].right;
    return nodes[id].value;
}
}

// trains on rows [0,n_train), tests on [n_test_begin,N). returns train seconds.
double train_exact(const Dataset& ds, const Params& p, int n_train, int n_test_begin,
                   double& rmse_train, double& rmse_test) {
    std::vector<int> rows(n_train);
    for (int i = 0; i < n_train; ++i) rows[i] = i;
    std::vector<ENode> nodes;
    double t0 = omp_get_wtime();
    grow_exact(ds, p, nodes, rows, 0);
    double t = omp_get_wtime() - t0;
    double se = 0;
    for (int i = 0; i < n_train; ++i) { double d = predict_exact(nodes, ds, i) - ds.y[i]; se += d * d; }
    rmse_train = std::sqrt(se / n_train);
    se = 0; int nt = ds.N - n_test_begin;
    for (int i = n_test_begin; i < ds.N; ++i) { double d = predict_exact(nodes, ds, i) - ds.y[i]; se += d * d; }
    rmse_test = std::sqrt(se / nt);
    return t;
}
