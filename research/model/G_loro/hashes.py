"""Write sha256 of PREREG_central_v3.md and every .py in ssl_central/ and ssl_central_v3/.
    python research/model/G_loro/hashes.py <out.txt>"""
import glob, hashlib, os, sys, datetime
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
files = ["research/model/PREREG_central_v3.md"] + sorted(
    os.path.relpath(f, R) for d in ("ssl_central", "ssl_central_v3") for f in glob.glob(os.path.join(R, d, "*.py")))
with open(sys.argv[1], "w") as fh:
    fh.write(f"# written {datetime.datetime.now().isoformat(timespec='seconds')}\n")
    for f in files:
        fh.write(f"{hashlib.sha256(open(os.path.join(R, f), 'rb').read()).hexdigest()}  {f}\n")
print(open(sys.argv[1]).read())
