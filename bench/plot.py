"""Make the report figures and a summary table from bench/final/*.csv.
Usage (repo root):  pip install pandas matplotlib ;  python bench/plot.py [bench/final]
Every number is the MEDIAN of the repeated runs; error bars span min..max.
"""
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = sys.argv[1] if len(sys.argv) > 1 else "bench/final"
FIG = os.path.join(D, "figs")
os.makedirs(FIG, exist_ok=True)
summary = []


def load(name):
    p = os.path.join(D, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def stat(s):
    return s.median(), s.min(), s.max()


def errbars(vals):
    med = [v[0] for v in vals]
    lo = [v[0] - v[1] for v in vals]
    hi = [v[2] - v[0] for v in vals]
    return med, [lo, hi]


# ---- 1. three-step story -------------------------------------------------------------
st = load("story.csv")
if st is not None:
    groups = [
        ("Exact\n(sort-based)", st[st["mode"] == "exact"]),
        ("Histogram\nserial, no subtraction", st[(st["mode"] == "hist") & (st["threads"] == 1) & (st["subtraction"] == 0)]),
        ("Histogram\nserial + subtraction", st[(st["mode"] == "hist") & (st["threads"] == 1) & (st["subtraction"] == 1)]),
        ("Histogram\nOpenMP (8 threads)", st[(st["mode"] == "hist") & (st["threads"] == 8)]),
    ]
    groups = [(n, g) for n, g in groups if len(g)]
    vals = [stat(g["train_time"]) for _, g in groups]
    med, err = errbars(vals)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar([n for n, _ in groups], med, yerr=err, capsize=4, color=["#999", "#6a9fd8", "#3b7bc4", "#1c4f94"][:len(groups)])
    ax.set_yscale("log")
    ax.set_ylabel("training time (s, log scale)")
    ax.set_title("Three-step story, N=1M, depth 6")
    for b, m in zip(bars, med):
        ax.text(b.get_x() + b.get_width() / 2, m * 1.08, f"{m:.3g}s", ha="center")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_story.png"), dpi=160)
    plt.close(fig)
    for (n, g), m in zip(groups, med):
        summary.append(("story", n.replace("\n", " "), m, med[0] / m, g["rmse_test"].median()))

# ---- 2. strong scaling + Amdahl ------------------------------------------------------------
sc = load("scaling.csv")
if sc is not None:
    t1 = sc[sc["threads"] == 1]
    T1 = t1["train_time"].median()
    H1 = t1["hist_time"].median()
    f = H1 / T1  # parallelizable fraction = histogram share of the serial run
    ths = sorted(sc["threads"].unique())
    rows = []
    for t in ths:
        g = sc[sc["threads"] == t]
        rows.append((t, g["train_time"].median(), g["train_time"].min(), g["hist_time"].median(),
                     g["hist_time"].min(), g["split_time"].median(), g["part_time"].median()))
    df = pd.DataFrame(rows, columns=["threads", "train", "train_min", "hist", "hist_min", "split", "part"])
    df["speedup"] = T1 / df["train"]
    df["hist_speedup"] = H1 / df["hist"]
    df["eff"] = df["speedup"] / df["threads"]
    amdahl = [1 / ((1 - f) + f / t) for t in ths]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot(ths, ths, "k--", label="ideal")
    ax[0].plot(ths, amdahl, "g:", label=f"Amdahl limit (parallel fraction f={f:.2f})")
    ax[0].plot(df["threads"], df["hist_speedup"], "s-", label="histogram step")
    ax[0].plot(df["threads"], df["speedup"], "o-", label="whole training")
    ax[0].set_xlabel("threads")
    ax[0].set_ylabel("speedup vs serial T1")
    ax[0].set_title("Strong scaling (N=1M, depth 6)")
    ax[0].legend()
    ax[0].grid(alpha=.3)
    ax[1].plot(df["threads"], df["eff"], "o-")
    ax[1].set_ylim(0, 1.05)
    ax[1].set_xlabel("threads")
    ax[1].set_ylabel("parallel efficiency (speedup / threads)")
    ax[1].set_title("Efficiency")
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_scaling.png"), dpi=160)
    plt.close(fig)

    # phase breakdown
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = [str(t) for t in df["threads"]]
    ax.bar(x, df["hist"], label="histogram (parallel)")
    ax.bar(x, df["part"], bottom=df["hist"], label="partition (serial)")
    ax.bar(x, df["split"], bottom=df["hist"] + df["part"], label="split search (serial)")
    ax.set_xlabel("threads")
    ax.set_ylabel("time (s)")
    ax.set_title("Where the time goes (median per phase)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_breakdown.png"), dpi=160)
    plt.close(fig)

    df.to_csv(os.path.join(D, "scaling_summary.csv"), index=False)
    print(f"T1 = {T1:.3f}s, histogram share f = {f:.2f}, Amdahl max speedup = {1/(1-f):.2f}x")
    print(df.round(3).to_string(index=False))
    for _, r in df.iterrows():
        summary.append(("scaling", f"{int(r.threads)} threads", r.train, r.speedup, float("nan")))

# ---- 3. cutoff -------------------------------------------------------------------------------
ct = load("cutoff.csv")
if ct is not None:
    cs = sorted(ct["cutoff"].unique())
    vals = [stat(ct[ct["cutoff"] == c]["hist_time"]) for c in cs]
    med, err = errbars(vals)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(range(len(cs)), med, yerr=err, fmt="o-", capsize=4)
    ax.set_xticks(range(len(cs)))
    ax.set_xticklabels([str(c) for c in cs])
    ax.set_xlabel("small-node cutoff (rows below this use the serial histogram)")
    ax.set_ylabel("histogram time (s)")
    ax.set_title("Small-node cutoff, 8 threads, depth 10")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_cutoff.png"), dpi=160)
    plt.close(fig)
    for c, m in zip(cs, med):
        summary.append(("cutoff", str(c), m, float("nan"), float("nan")))

# ---- 4. schedule, padding, parallel split (grouped comparisons) ----------------------------
def compare(fname, col, label, title, out, metric):
    d = load(fname)
    if d is None:
        return
    ks = list(d[col].unique())
    vals = [stat(d[d[col] == k][metric]) for k in ks]
    med, err = errbars(vals)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.bar([str(k) for k in ks], med, yerr=err, capsize=5, color="#3b7bc4")
    ax.set_ylabel(f"{metric.replace('_', ' ')} (s)")
    ax.set_xlabel(label)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, out), dpi=160)
    plt.close(fig)
    for k, m in zip(ks, med):
        summary.append((fname.replace(".csv", ""), f"{label}={k}", m, float("nan"), float("nan")))


compare("sched.csv", "schedule", "schedule", "Static vs dynamic (8 threads)", "fig5_schedule.png", "hist_time")
compare("padding.csv", "pad", "pad (1=64B aligned)", "Aligned vs misaligned private histograms", "fig6_padding.png", "hist_time")
compare("parsplit.csv", "par_split", "par_split", "Parallel split search, depth 10", "fig7_parsplit.png", "split_time")

pd.DataFrame(summary, columns=["experiment", "setting", "median_time_s", "speedup", "rmse_test"]).to_csv(
    os.path.join(D, "summary.csv"), index=False)
print(f"figures in {FIG}, table in {os.path.join(D, 'summary.csv')}")