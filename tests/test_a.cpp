#include "common.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <set>
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s\n", msg); fails++; } else printf("PASS: %s\n", msg); } while (0)

int main() {
    int fails = 0;
    // Test 1: section 5 table
    { Hist h(256); int cnt[4] = {2,2,2,2}; double sm[4] = {22,24,62,64};
      for (int b = 0; b < 4; ++b) { h[b].cnt = cnt[b]; h[b].sum_y = sm[b]; }
      Split s = find_best_split(h, 1, 1);
      CHECK(s.bin == 1 && std::fabs(s.gain - 800) < 1e-9, "section 5 table: best bin 1, gain 800"); }

    // Test 2: root of 8-row dataset
    Dataset t = make_tiny();
    std::vector<int> all(8); for (int i = 0; i < 8; ++i) all[i] = i;
    Hist root = build_hist(all, t);
    { Split s = find_best_split(root, 2, 1);
      CHECK(s.feature == 0 && s.bin == 1 && std::fabs(s.gain - 1682) < 1e-9, "8-row root: feature A, bin 1, gain 1682");
      Split sb = best_for_feature(root, 1, 1);
      CHECK(std::fabs(sb.gain - 242) < 1e-9, "8-row root: feature B alone gain 242"); }

    // Test 3: full 8-row tree
    Params p; p.max_depth = 2; p.min_leaf = 1; p.min_gain = 10;
    for (int sub = 0; sub < 2; ++sub) {
        p.use_subtraction = sub;
        std::vector<TreeNode> nodes; grow(t, p, nodes, all, root, 0);
        bool ok = nodes.size() == 5 && nodes[0].feature == 0 && nodes[0].bin == 1;
        int R = nodes[0].right, L = nodes[0].left;
        ok = ok && nodes[R].feature == 1 && nodes[R].bin == 0 && std::fabs(nodes[L].value - 11.5) < 1e-9
                && std::fabs(nodes[nodes[R].left].value - 30.5) < 1e-9 && std::fabs(nodes[nodes[R].right].value - 50.5) < 1e-9;
        CHECK(ok, sub ? "8-row tree shape + leaves (subtraction ON)" : "8-row tree shape + leaves (subtraction OFF)");
        Dataset q = t; q.N = 9; q.X_bin.push_back(3); q.X_bin.push_back(1); q.y.push_back(0);
        CHECK(std::fabs(predict_bin(nodes, q, 8) - 50.5) < 1e-9, "predict A=3,B=1 -> 50.5");
    }

    // Test 4: parallel split == serial split on random histogram
    { std::mt19937 rng(1); int D = 50; Hist h((size_t)D * 256);
      for (int f = 0; f < D; ++f) for (int i = 0; i < 5000; ++i) { Bin& b = h[f*256 + rng()%256]; b.cnt++; b.sum_y += (rng()%1000)/10.0; }
      Split a = find_best_split(h, D, 20), b = find_best_split_omp(h, D, 20);
      CHECK(a.feature == b.feature && a.bin == b.bin && a.gain == b.gain, "parallel split search == serial"); }

    // Test 5: binning on small synthetic data (needs data/d100k.bin)
    { Dataset ds = load_raw("../data/d100k.bin"); bin_dataset(ds, 256);
      CHECK(ds.X_bin.size() == (size_t)ds.N * ds.D, "X_bin is exactly N*D bytes");
      std::set<int> s20; for (int i = 0; i < ds.N; ++i) s20.insert(ds.X_bin[(size_t)i*ds.D + 20]);
      CHECK(s20.size() == 10, "feature 20 uses exactly 10 bins (lossless)");
      std::set<int> s0; for (int i = 0; i < ds.N; ++i) s0.insert(ds.X_bin[(size_t)i*ds.D + 0]);
      CHECK(s0.size() == 256 || s0.size() == 255, "feature 0 uses ~256 quantile bins"); }

    // Test 6: subtraction on/off on real data
    { Dataset ds = load_raw("../data/d100k.bin"); bin_dataset(ds, 256);
      std::vector<int> rows(80000); for (int i = 0; i < 80000; ++i) rows[i] = i;
      Hist r = build_hist(rows, ds); Params q;
      double rm[2]; size_t nn[2];
      for (int sub = 0; sub < 2; ++sub) { q.use_subtraction = sub; std::vector<TreeNode> nd; grow(ds, q, nd, rows, r, 0);
        double se = 0; for (int i = 0; i < 80000; ++i) { double d = predict_bin(nd, ds, i) - ds.y[i]; se += d*d; }
        rm[sub] = std::sqrt(se / 80000); nn[sub] = nd.size(); }
      printf("   rmse off=%.12f on=%.12f nodes %zu/%zu\n", rm[0], rm[1], nn[0], nn[1]);
      CHECK(nn[0] == nn[1] && std::fabs(rm[0] - rm[1]) < 1e-9, "subtraction ON/OFF: same tree size and RMSE"); }

    printf("\n%s (%d failures)\n", fails ? "SOME TESTS FAILED" : "ALL TESTS PASSED", fails);
    return fails;
}
