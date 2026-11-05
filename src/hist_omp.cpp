// Parallel histogram build: every thread fills its OWN private histogram over a share of the
// rows (no atomics, no sharing), then the private copies are summed bin-by-bin in parallel.
#include "common.h"
#include <omp.h>
#include <cstdint>
#include <cstring>

static_assert(sizeof(Bin) == 16, "Bin is expected to be 16 bytes (4 per 64-byte cache line)");

namespace
{
    // One private histogram. The data pointer is aligned to a 64-byte cache line so that two
    // threads never write into the same line, whatever the allocator returned.
    struct PrivHist
    {
        std::vector<Bin> storage;
        Bin *ptr = nullptr;
        size_t n = 0;
        void ensure(size_t nbins)
        {
            if (n == nbins)
                return;
            storage.assign(nbins + 4, Bin()); // 4 extra bins (64 B) of slack for alignment
            uintptr_t a = ((uintptr_t)storage.data() + 63) & ~(uintptr_t)63;
            ptr = (Bin *)a;
            n = nbins;
        }
    };
    std::vector<PrivHist> g_pool; // allocated once, reused by every node (avoids 127 x T allocations)
} // namespace

Hist build_hist_omp(const Dataset &ds, const std::vector<int> &rows, const Params &p)
{
    const int T = p.threads;
    // small nodes: thread start-up + reduction cost more than they save
    if (T <= 1 || (int)rows.size() < p.omp_cutoff)
        return build_hist(ds, rows);

    const int D = ds.D;
    const size_t nb = (size_t)D * NBINS;
    if ((int)g_pool.size() < T)
        g_pool.resize(T);
    for (int t = 0; t < T; ++t)
        g_pool[t].ensure(nb);

    Hist out(nb);
    const int n = (int)rows.size();
    const int *r = rows.data();
    const uint8_t *X = ds.X_bin.data();
    const float *y = ds.y.data();

    omp_set_schedule(p.dynamic_sched ? omp_sched_dynamic : omp_sched_static, p.dynamic_sched ? 4096 : 0);

#pragma omp parallel num_threads(T)
    {
        const int tid = omp_get_thread_num();
        const int nt = omp_get_num_threads(); // may be < T if the runtime gave fewer
        Bin *h = g_pool[tid].ptr;
        std::memset((void *)h, 0, nb * sizeof(Bin)); // each thread clears its own copy (first touch)

        // 1) accumulate into the private histogram
#pragma omp for schedule(runtime)
        for (int k = 0; k < n; ++k)
        {
            const int i = r[k];
            const uint8_t *xb = X + (size_t)i * D;
            const double yy = y[i];
            for (int f = 0; f < D; ++f)
            {
                Bin &b = h[(size_t)f * NBINS + xb[f]];
                b.sum_y += yy;
                b.cnt++;
            }
        } // implicit barrier: all private histograms are complete

        // 2) reduction: bins are split among threads, each sums nt copies of its bins
#pragma omp for schedule(static)
        for (long long j = 0; j < (long long)nb; ++j)
        {
            double s = 0;
            uint32_t c = 0;
            for (int t = 0; t < nt; ++t)
            {
                s += g_pool[t].ptr[j].sum_y;
                c += g_pool[t].ptr[j].cnt;
            }
            out[j].sum_y = s;
            out[j].cnt = c;
        }
    }
    return out;
}