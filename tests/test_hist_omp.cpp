// build_hist_omp must equal build_hist: counts exactly, sums up to floating-point reordering.
#include "common.h"
#include <cmath>
#include <cstdio>
#include <random>

int main()
{
    Dataset ds;
    ds.N = 200000;
    ds.D = 50;
    std::mt19937 g(1);
    ds.X_bin.resize((size_t)ds.N * ds.D);
    ds.y.resize(ds.N);
    for (auto &v : ds.X_bin)
        v = (uint8_t)(g() & 255);
    for (auto &v : ds.y)
        v = (float)(g() % 1000) / 37.0f;

    std::vector<int> all(ds.N), half, tiny;
    for (int i = 0; i < ds.N; ++i)
        all[i] = i;
    for (int i = 0; i < ds.N; ++i)
        if (g() & 1)
            half.push_back(i); // ascending, like a real node
    for (int i = 0; i < 100; ++i)
        tiny.push_back(i * 7);                               // below the cutoff
    std::vector<int> odd(all.begin(), all.begin() + 100003); // size not divisible by thread count

    int bad = 0, runs = 0;
    for (const auto *rows : {&all, &half, &tiny, &odd})
    {
        Hist ref = build_hist(ds, *rows);
        for (int T : {1, 2, 3, 4, 8})
            for (int dyn = 0; dyn < 2; ++dyn)
                for (int pad = 0; pad < 2; ++pad)
                {
                    Params p;
                    p.threads = T;
                    p.dynamic_sched = dyn;
                    p.pad = pad;
                    p.omp_cutoff = 2048;
                    Hist h = build_hist_omp(ds, *rows, p);
                    ++runs;
                    for (size_t j = 0; j < ref.size(); ++j)
                        if (h[j].cnt != ref[j].cnt ||
                            std::fabs(h[j].sum_y - ref[j].sum_y) > 1e-9 * (1 + std::fabs(ref[j].sum_y)))
                        {
                            printf("MISMATCH rows=%zu T=%d dyn=%d bin=%zu\n", rows->size(), T, dyn, j);
                            ++bad;
                            break;
                        }
                }
    }
    printf("%d/%d configurations match build_hist\n", runs - bad, runs);
    puts(bad ? "FAILED" : "ALL TESTS PASSED");
    return bad != 0;
}