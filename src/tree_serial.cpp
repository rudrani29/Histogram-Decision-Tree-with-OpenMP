#include "tree.h"
#include <cmath>
#include <cstdio>
#include <numeric>
#include <omp.h>

TrainStats g_stats;

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

// Sum of y and count over bins [lo, hi] of one feature's histogram.
static void range_stats(const Hist &h, int f, int lo, int hi, double &S, uint32_t &n)
{
    S = 0;
    n = 0;
    const Bin *hf = &h[(size_t)f * NBINS];
    for (int b = lo; b <= hi; ++b)
    {
        S += hf[b].sum_y;
        n += hf[b].cnt;
    }
}

// Recursive node expansion. Returns the index of the node it created.
// h      : this node's histogram. May be nullptr only if depth >= max_depth (a forced leaf).
// S, n   : sum of y and row count of this node (known from the parent's split, so a
//          node that is a forced leaf never needs its own histogram).
// NOTE: always use indices, never keep a reference into t.nodes across a
// recursive call (emplace_back may reallocate the vector).
static int grow(Tree &t, const Dataset &ds, const std::vector<int> &rows, const Hist *h,
                double S, uint32_t n, int depth, const Params &p, const HistFn &hist_fn)
{
    int idx = (int)t.nodes.size();
    t.nodes.emplace_back();
    t.nodes[idx].depth = depth;
    t.nodes[idx].value = n ? (float)(S / n) : 0.f;

    // stop conditions
    if (depth >= p.max_depth || (int)n < 2 * p.min_leaf)
        return idx;

    // best split (A's functions)
    double t0 = omp_get_wtime();
    Split s = p.par_split ? find_best_split_omp(*h, ds.D, p.min_leaf)
                          : find_best_split(*h, ds.D, p.min_leaf);
    g_stats.split_s += omp_get_wtime() - t0;
    if (s.feature < 0 || s.gain < p.min_gain)
        return idx;

    // divide rows
    t0 = omp_get_wtime();
    std::vector<int> L, R;
    partition(ds, rows, s.feature, s.bin, L, R);
    g_stats.part_s += omp_get_wtime() - t0;
    if (L.empty() || R.empty())
        return idx;

    // child statistics straight from the parent's histogram (no extra pass over rows)
    double SL, SR;
    uint32_t nL, nR;
    range_stats(*h, s.feature, 0, s.bin, SL, nL);
    range_stats(*h, s.feature, s.bin + 1, NBINS - 1, SR, nR);

    // child histograms: only needed if a child can still split
    Hist hl, hr;
    bool child_leaf = (depth + 1 >= p.max_depth);
    if (!child_leaf)
    {
        t0 = omp_get_wtime();
        if (p.use_subtraction)
        {
            // build the smaller child, get the larger one as parent - small
            bool left_small = L.size() <= R.size();
            Hist small = hist_fn(ds, left_small ? L : R);
            Hist large = subtract(*h, small);
            hl = left_small ? std::move(small) : std::move(large);
            hr = left_small ? std::move(large) : std::move(small);
        }
        else
        {
            hl = hist_fn(ds, L);
            hr = hist_fn(ds, R);
        }
        g_stats.hist_s += omp_get_wtime() - t0;
    }

    t.nodes[idx].feature = s.feature;
    t.nodes[idx].bin = s.bin;
    int l = grow(t, ds, L, child_leaf ? nullptr : &hl, SL, nL, depth + 1, p, hist_fn);
    int r = grow(t, ds, R, child_leaf ? nullptr : &hr, SR, nR, depth + 1, p, hist_fn);
    t.nodes[idx].left = l;
    t.nodes[idx].right = r;
    return idx;
}

Tree train_tree(const Dataset &ds, const Params &p, HistFn hist_fn, int n_train)
{
    if (n_train < 0 || n_train > ds.N)
        n_train = ds.N;
    g_stats = TrainStats();

    Tree t;
    std::vector<int> rows(n_train);
    std::iota(rows.begin(), rows.end(), 0);

    // root histogram (always built; it is the only node with no parent to subtract from)
    double t0 = omp_get_wtime();
    Hist root = hist_fn(ds, rows);
    g_stats.hist_s += omp_get_wtime() - t0;

    double S;
    uint32_t n;
    range_stats(root, 0, 0, NBINS - 1, S, n); // every row falls in exactly one bin of feature 0
    grow(t, ds, rows, &root, S, n, 0, p, hist_fn);
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

double rmse_range(const Tree &t, const Dataset &ds, int begin, int end)
{
    double s = 0;
    for (int i = begin; i < end; ++i)
    {
        double d = (double)predict_row(t, ds, i) - ds.y[i];
        s += d * d;
    }
    return std::sqrt(s / (end - begin));
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