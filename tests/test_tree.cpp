// Build (one line): g++ -O2 -fopenmp -std=c++17 -Isrc tests/test_tree.cpp src/hist.cpp src/tree_serial.cpp src/split_tmp.cpp -o test_tree
#include "tree.h"
#include <cstdio>
#include <cmath>

static int fails = 0;
#define CHECK(c)                                          \
    do                                                    \
    {                                                     \
        if (!(c))                                         \
        {                                                 \
            printf("FAIL: %s (line %d)\n", #c, __LINE__); \
            ++fails;                                      \
        }                                                 \
    } while (0)

int main()
{
    // ---- Test 1: section 5 histogram (1 feature, 8 rows) ----
    {
        Dataset ds;
        ds.N = 8;
        ds.D = 1;
        ds.X_bin = {0, 0, 1, 1, 2, 2, 3, 3};
        ds.y = {10, 12, 11, 13, 30, 32, 31, 33};
        std::vector<int> rows = {0, 1, 2, 3, 4, 5, 6, 7};
        Hist h = build_hist(ds, rows);
        uint32_t cnt[4] = {2, 2, 2, 2};
        double sum[4] = {22, 24, 62, 64};
        for (int b = 0; b < 4; ++b)
        {
            CHECK(h[b].cnt == cnt[b]);
            CHECK(h[b].sum_y == sum[b]);
        }
        Split s = find_best_split(h, 1, 1);
        CHECK(s.feature == 0 && s.bin == 1 && std::fabs(s.gain - 800) < 1e-9);
        printf("Test 1 (histogram + best split): done\n");
    }
    // ---- Test 2: section 11 full tree (features A, B) ----
    {
        Dataset ds;
        ds.N = 8;
        ds.D = 2;
        ds.X_bin = {0, 0, 0, 1, 1, 0, 1, 1, 2, 0, 2, 1, 3, 0, 3, 1}; // row-major [A,B]
        ds.y = {10, 12, 11, 13, 30, 50, 31, 51};
        Params p;
        p.max_depth = 2;
        p.min_leaf = 1;
        p.min_gain = 10;
        Tree t = train_tree(ds, p, build_hist);
        print_tree(t);
        CHECK(t.nodes[0].feature == 0 && t.nodes[0].bin == 1); // A <= 1
        CHECK(t.nodes.size() == 5);
        // new row A=3, B=1
        Dataset q;
        q.N = 1;
        q.D = 2;
        q.X_bin = {3, 1};
        q.y = {0};
        float pr = predict_row(t, q, 0);
        printf("Prediction for A=3,B=1: %.2f (expected 50.5)\n", pr);
        CHECK(std::fabs(pr - 50.5f) < 1e-4);
        auto pred = predict_all(t, ds);
        printf("Train RMSE: %.4f\n", rmse(pred, ds.y));
    }
    printf(fails ? "\n%d CHECK(S) FAILED\n" : "\nALL TESTS PASSED\n", fails);
    return fails != 0;
}