"""T6: two real-system runs with the same configuration give bit-identical weights. T3: every file a client process opened
lies in its own client folder, the code (fl_system, ssl_central), the Python environments or the system; no other data file.
    .venv-fl/bin/python fl_system/tests/test_T6_T3.py <run_dir_a> <run_dir_b(traced)> <fold_dir>"""
import glob, json, os, sys, time, torch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
a, b, fold = [os.path.abspath(x) for x in sys.argv[1:4]]; R = {}
fa, fb = torch.load(os.path.join(a, "final.pt")), torch.load(os.path.join(b, "final.pt"))
R["T6_final_maxabs"] = max(float((fa[m][k] - fb[m][k]).abs().max()) for m in ("enc", "head") for k in fa[m])
R["T6_rounds_equal"] = all(torch.equal(torch.load(os.path.join(a, f"global_r{r:03d}.pt")), torch.load(os.path.join(b, f"global_r{r:03d}.pt"))) for r in (1, 2, 3))
R["T6_PASS"] = R["T6_final_maxabs"] == 0.0 and R["T6_rounds_equal"]
DATA = os.path.join(ROOT, "data"); viol = {}; n_files = {}
for t in sorted(glob.glob(os.path.join(b, "trace_client_*.txt"))):
    k = int(os.path.basename(t)[13:15]); own = os.path.join(fold, f"client_{k:02d}")
    files = [l for l in open(t).read().splitlines() if l]; n_files[k] = len(files)
    bad = [f for f in files if f.startswith(DATA) and not f.startswith(own + os.sep)]     # any data file outside the own folder
    bad += [f for f in files if f.startswith(os.path.join(ROOT, "research")) or f.startswith(os.path.join(ROOT, "reports"))]
    if bad: viol[k] = bad[:10]
R["T3_files_opened_per_client"] = n_files; R["T3_violations"] = viol; R["T3_PASS"] = len(n_files) == 10 and not viol
own_data = {k: sorted({os.path.basename(f) for f in open(os.path.join(b, f"trace_client_{k:02d}.txt")).read().splitlines() if f.startswith(DATA)}) for k in n_files}
R["T3_data_files_opened_example"] = own_data.get(3)
out = os.path.join(ROOT, "fl_system/tests", time.strftime("T6_T3_%Y%m%d_%H%M%S.json")); json.dump(R, open(out, "x"), indent=1); print(json.dumps(R)[:900], "->", out)
