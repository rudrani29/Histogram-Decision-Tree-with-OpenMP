// STAND-IN for Person B's code so A can test. Replace/remove when B's tree is merged.
#include "common.h"

Hist build_hist(const std::vector<int>& rows, const Dataset& ds) {
    Hist h((size_t)ds.D * 256);
    for (int i : rows) {
        const uint8_t* xb = &ds.X_bin[(size_t)i * ds.D];
        for (int f = 0; f < ds.D; ++f) {
            Bin& b = h[(size_t)f * 256 + xb[f]];
            b.sum_y += ds.y[i]; b.cnt++;
        }
    }
    return h;
}

int grow(const Dataset& ds, const Params& p, std::vector<TreeNode>& nodes,
         std::vector<int> rows, const Hist& hist, int depth) {
    int id = nodes.size(); nodes.emplace_back();
    double S = 0; uint32_t n = 0;
    for (int b = 0; b < 256; ++b) { S += hist[b].sum_y; n += hist[b].cnt; }
    nodes[id].value = n ? S / n : 0;
    if (depth >= p.max_depth || n < 2u * (uint32_t)p.min_leaf) return id;

    Split s = p.par_split ? find_best_split_omp(hist, ds.D, p.min_leaf)
                          : find_best_split(hist, ds.D, p.min_leaf);
    if (s.feature < 0 || s.gain < p.min_gain) return id;

    std::vector<int> L, R;
    for (int i : rows) (ds.X_bin[(size_t)i * ds.D + s.feature] <= s.bin ? L : R).push_back(i);

    Hist hl, hr;
    if (p.use_subtraction) {
        bool leftSmall = L.size() <= R.size();
        Hist small = build_hist(leftSmall ? L : R, ds);
        Hist large = subtract(hist, small);
        hl = leftSmall ? std::move(small) : std::move(large);
        hr = leftSmall ? std::move(large) : std::move(small);
    } else {
        hl = build_hist(L, ds);
        hr = build_hist(R, ds);
    }
    nodes[id].feature = s.feature; nodes[id].bin = s.bin;
    int l = grow(ds, p, nodes, std::move(L), hl, depth + 1);
    int r = grow(ds, p, nodes, std::move(R), hr, depth + 1);
    nodes[id].left = l; nodes[id].right = r;
    return id;
}

double predict_bin(const std::vector<TreeNode>& nodes, const Dataset& ds, int row) {
    int id = 0;
    while (nodes[id].left >= 0)
        id = ds.X_bin[(size_t)row * ds.D + nodes[id].feature] <= nodes[id].bin ? nodes[id].left : nodes[id].right;
    return nodes[id].value;
}
