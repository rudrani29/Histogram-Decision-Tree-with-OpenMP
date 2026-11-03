#include "tree.h"
#include <cmath>
#include <cstdio>
#include <numeric>

void partition(const Dataset &ds, const std::vector<int> &rows, int feature, int bin,
               std::vector<int> &left, std::vector<int> &right)
{
    left.clear();
    right.clear();
    for (int i : rows)
    {
        if (ds.X_bin[(size_t)i * ds.D + feature] <= bin)
            left.push_back(i);
        else
            right.push_back(i);
    }
}

// Recursive node expansion. Returns the index of the node it created.
// NOTE: always use indices, never keep a reference into t.nodes across a
// recursive call (push_back may reallocate the vector).
static int grow(Tree &t, const Dataset &ds, const std::vector<int> &rows,
                int depth, const Params &p, const HistFn &hist_fn)
{
    int idx = (int)t.nodes.size();
    t.nodes.emplace_back();
    t.nodes[idx].depth = depth;

    // 1. histogram for this node (A will add the subtraction trick here later)
    Hist h = hist_fn(ds, rows);

    // node mean: sum over all bins of feature 0 gives total sum_y and count
    double S = 0;
    uint32_t n = 0;
    for (int b = 0; b < 256; ++b)
    {
        S += h[b].sum_y;
        n += h[b].cnt;
    }
    t.nodes[idx].value = n ? (float)(S / n) : 0.f;

    // 2. stop conditions
    if (depth >= p.max_depth || (int)n < 2 * p.min_leaf)
        return idx;

    // 3. best split (A's function)
    Split s = find_best_split(h, ds.D, p.min_leaf);
    if (s.feature < 0 || s.gain < p.min_gain)
        return idx;

    // 4. divide rows, recurse
    std::vector<int> L, R;
    partition(ds, rows, s.feature, s.bin, L, R);
    if (L.empty() || R.empty())
        return idx;

    t.nodes[idx].feature = s.feature;
    t.nodes[idx].bin = s.bin;
    int l = grow(t, ds, L, depth + 1, p, hist_fn);
    int r = grow(t, ds, R, depth + 1, p, hist_fn);
    t.nodes[idx].left = l;
    t.nodes[idx].right = r;
    return idx;
}

Tree train_tree(const Dataset &ds, const Params &p, HistFn hist_fn)
{
    Tree t;
    std::vector<int> rows(ds.N);
    std::iota(rows.begin(), rows.end(), 0);
    grow(t, ds, rows, 0, p, hist_fn);
    return t;
}

float predict_row(const Tree &t, const Dataset &ds, int row)
{
    int k = 0;
    while (!t.nodes[k].is_leaf())
    {
        const Node &nd = t.nodes[k];
        k = (ds.X_bin[(size_t)row * ds.D + nd.feature] <= nd.bin) ? nd.left : nd.right;
    }
    return t.nodes[k].value;
}

std::vector<float> predict_all(const Tree &t, const Dataset &ds)
{
    std::vector<float> out(ds.N);
    for (int i = 0; i < ds.N; ++i)
        out[i] = predict_row(t, ds, i);
    return out;
}

double rmse(const std::vector<float> &pred, const std::vector<float> &y)
{
    double s = 0;
    for (size_t i = 0; i < y.size(); ++i)
    {
        double d = (double)pred[i] - y[i];
        s += d * d;
    }
    return std::sqrt(s / y.size());
}

void print_tree(const Tree &t, int k, int indent)
{
    const Node &nd = t.nodes[k];
    for (int i = 0; i < indent; ++i)
        printf("  ");
    if (nd.is_leaf())
        printf("Leaf: %.4f\n", nd.value);
    else
    {
        printf("[f%d <= %d]  (mean %.4f)\n", nd.feature, nd.bin, nd.value);
        print_tree(t, nd.left, indent + 1);
        print_tree(t, nd.right, indent + 1);
    }
}