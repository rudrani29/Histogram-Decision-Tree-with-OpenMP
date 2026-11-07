# Run from the repo root:  powershell -ExecutionPolicy Bypass -File bench/run_all.ps1
# Setup: plugged in, OneDrive paused, other apps closed.  Everything is pinned to the 8 P-core
# hardware threads (affinity mask FF) with passive waiting, and every experiment interleaves its
# settings round by round so drift hits all of them equally.
$ErrorActionPreference = "Continue"
$out = "bench/final"
New-Item -ItemType Directory -Force $out | Out-Null
Remove-Item "$out/*.csv" -ErrorAction SilentlyContinue
$env:OMP_WAIT_POLICY = "passive"
$d1 = "data/d1m.bin"

function Run([string]$csv, [string[]]$a) {
    cmd /c start /wait /b /affinity FF hist_tree.exe @a --out "$out/$csv" | Out-Null
}

# 1. Three-step story (1M rows, depth 6): exact -> serial hist (no subtraction) -> serial hist + subtraction -> OpenMP
Write-Host "== story"
1..3 | % {
    Run "story.csv" @("--data", $d1, "--mode", "exact")
}
1..7 | % {
    Run "story.csv" @("--data", $d1, "--threads", "1", "--histomp", "0", "--subtraction", "0")
    Run "story.csv" @("--data", $d1, "--threads", "1", "--histomp", "0", "--subtraction", "1")
    Run "story.csv" @("--data", $d1, "--threads", "8", "--histomp", "1", "--subtraction", "1", "--cutoff", "8192")
}

# 2. Strong scaling (1M rows, depth 6)
Write-Host "== scaling"
1..7 | % {
    Run "scaling.csv" @("--data", $d1, "--threads", "1", "--histomp", "0")
    foreach ($t in 2, 4, 6, 8) {
        Run "scaling.csv" @("--data", $d1, "--threads", "$t", "--histomp", "1", "--cutoff", "8192")
    }
}

# 3. Small-node cutoff (depth 10 so that there are many small nodes)
Write-Host "== cutoff"
1..5 | % {
    foreach ($c in 0, 512, 2048, 8192, 32768) {
        Run "cutoff.csv" @("--data", $d1, "--threads", "8", "--histomp", "1", "--depth", "10", "--cutoff", "$c")
    }
}

# 4. Scheduling and 5. padding (8 threads)
Write-Host "== schedule + padding"
1..10 | % {
    foreach ($s in "static", "dynamic") {
        Run "sched.csv" @("--data", $d1, "--threads", "8", "--histomp", "1", "--cutoff", "8192", "--schedule", $s)
    }
    foreach ($pd in 1, 0) {
        Run "padding.csv" @("--data", $d1, "--threads", "8", "--histomp", "1", "--cutoff", "8192", "--pad", "$pd")
    }
}

# 6. Parallel split search at depth 10 (A's feature, shown with B's histogram)
Write-Host "== parsplit"
1..5 | % {
    foreach ($ps in 0, 1) {
        Run "parsplit.csv" @("--data", $d1, "--threads", "8", "--histomp", "1", "--cutoff", "8192", "--depth", "10", "--parsplit", "$ps")
    }
}

Remove-Item Env:OMP_WAIT_POLICY -ErrorAction SilentlyContinue
Write-Host "done. Now run: python bench/plot.py"