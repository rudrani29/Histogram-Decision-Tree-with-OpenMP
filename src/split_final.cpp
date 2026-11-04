#include "common.h"
#include <algorithm>

Split best_for_feature(const Hist& h, int f, int min_leaf) {
    const Bin* hf = &h[f * 256];
    uint32_t ml = std::max(1, min_leaf);
    double S = 0; uint32_t n = 0;
    for (int b = 0; b < 256; ++b) { S += hf[b].sum_y; n += hf[b].cnt; }
    Split best;
    if (n == 0) return best;
    double base = S * S / n, S_L = 0; uint32_t n_L = 0;
    for (int b = 0; b < 255; ++b) {
        S_L += hf[b].sum_y; n_L += hf[b].cnt;
        uint32_t n_R = n - n_L;
        if (n_L < ml || n_R < ml) continue;
        double S_R = S - S_L;
        double gain = S_L * S_L / n_L + S_R * S_R / n_R - base;
        if (gain > best.gain) best = {gain, f, b};
    }
    return best;
}

Split find_best_split(const Hist& h, int D, int min_leaf) {
    Split best;
    for (int f = 0; f < D; ++f) {
        Split s = best_for_feature(h, f, min_leaf);
        if (s.gain > best.gain) best = s;
    }
    return best;
}

Split find_best_split_omp(const Hist& h, int D, int min_leaf) {
    std::vector<Split> per(D);
    #pragma omp parallel for schedule(static)
    for (int f = 0; f < D; ++f) per[f] = best_for_feature(h, f, min_leaf);
    Split best;
    for (int f = 0; f < D; ++f) if (per[f].gain > best.gain) best = per[f];
    return best;
}

Hist subtract(const Hist& parent, const Hist& small) {
    Hist large(parent.size());
    for (size_t j = 0; j < parent.size(); ++j) {
        large[j].sum_y = parent[j].sum_y - small[j].sum_y;
        large[j].cnt   = parent[j].cnt   - small[j].cnt;
    }
    return large;
}
