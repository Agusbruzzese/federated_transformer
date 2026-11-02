"""
research/audit/auditor.py
=========================
Independent, low-cost external auditor for the extraction re-design.

It sends one workstream's REPORT.md + code (or the orchestrator's log) to Claude
with a skeptical-reviewer prompt, and writes the verdict next to the work as
AUDIT_<timestamp>.md. Usage and estimated cost are appended to usage.jsonl.

Usage
-----
    .venv/bin/python research/audit/auditor.py research/extraction/A_integrity
    .venv/bin/python research/audit/auditor.py orchestrator
    .venv/bin/python research/audit/auditor.py research/extraction/C_detection --model claude-haiku-4-5

API key: ANTHROPIC_API_KEY, else ~/.config/federated_transformer/anthropic_api_key
(kept outside the repo on purpose; never commit it).
"""

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import anthropic

ROOT = Path(__file__).resolve().parents[2]
EXTRACTION = ROOT / "research" / "extraction"
USAGE_LOG = Path(__file__).resolve().parent / "usage.jsonl"
KEY_FILE = Path.home() / ".config" / "federated_transformer" / "anthropic_api_key"

DEFAULT_MODEL = "claude-sonnet-5-5"
PRICES = {  # USD per 1M tokens (input, output)
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0),
}
CONTEXT_CHAR_BUDGET = 300_000  # ~75k tokens; files beyond this are listed, not sent
TEXT_SUFFIXES = {".md", ".py", ".csv", ".json", ".txt"}
MAX_DATA_FILE_CHARS = 8_000    # csv/json/txt results larger than this are listed only

SYSTEM = """You are an independent external auditor for a scientific signal-processing project.
The project: federated contrastive learning on an unlabeled single-channel TMR spintronic
magnetic sensor (2400 Hz, 14 days) that records passing vehicles. The feature-extraction stage is
being re-designed by several AI research agents coordinated by an orchestrator. You did not do the
work and have no stake in it. Your job is to find what is wrong, weak, or unsupported.

Audit for:
1. Brief compliance: which questions in the brief were answered, partly answered, or skipped.
2. Claims vs evidence: every quantitative claim must be backed by a method and a number that the
   code could plausibly produce. Flag overclaiming, missing uncertainty, and confidence levels that
   don't match the evidence.
3. Methodology: sampling bias (e.g. few days or hours), circular evaluation (evaluating a method on
   data it selected or was tuned on), leakage, criteria not pre-registered or changed after seeing
   results, unfair baselines, wrong statistics.
4. Code: bugs that would change conclusions (units, off-by-one, dtype or endianness, filter phase,
   edge effects, wrong axis, silent NaN handling). Cite the file and function.
5. Reproducibility: seeds, fixed file lists, whether the scripts would regenerate the numbers.
6. Cross-workstream consistency, when context from other workstreams is given.

Be concise and concrete; do not praise. Output markdown with exactly these sections:
## Verdict
One of ACCEPT / ACCEPT WITH FIXES / REJECT, plus one sentence.
## Issues
A numbered list; each item is `[BLOCKER|MAJOR|MINOR] <file or section>: <problem> -> <required fix>`.
## Brief coverage
One line per brief question: answered / partial / missing.
## Required follow-ups
Concrete checks or experiments, cheapest first."""


def load_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    if KEY_FILE.exists():
        return KEY_FILE.read_text().strip()
    sys.exit(f"No API key: set ANTHROPIC_API_KEY or create {KEY_FILE}")


def collect(target: Path) -> tuple[str, list[str]]:
    """Concatenate the target's text files within budget; return (bundle, omitted list)."""
    files = sorted(p for p in target.rglob("*") if p.is_file())
    # Report first, then code, then small result files
    order = {".md": 0, ".py": 1}
    files.sort(key=lambda p: (order.get(p.suffix, 2), p.stat().st_size))

    parts, omitted, used = [], [], 0
    for p in files:
        rel = p.relative_to(ROOT)
        if p.suffix not in TEXT_SUFFIXES or p.name.startswith("AUDIT_"):
            omitted.append(f"{rel} ({p.stat().st_size} B, not text or prior audit)")
            continue
        text = p.read_text(errors="replace")
        if p.suffix in {".csv", ".json", ".txt"} and len(text) > MAX_DATA_FILE_CHARS:
            omitted.append(f"{rel} ({len(text)} chars, data file too large)")
            continue
        if used + len(text) > CONTEXT_CHAR_BUDGET:
            omitted.append(f"{rel} ({len(text)} chars, over context budget)")
            continue
        parts.append(f'<file path="{rel}">\n{text}\n</file>')
        used += len(text)
    return "\n\n".join(parts), omitted


def build_request(target_arg: str) -> tuple[str, Path]:
    briefs = (EXTRACTION / "BRIEFS.md").read_text()

    if target_arg == "orchestrator":
        out_dir = EXTRACTION
        work = []
        for name in ("ORCHESTRATION_LOG.md", "SYNTHESIS.md"):
            p = EXTRACTION / name
            if p.exists():
                work.append(f'<file path="{p.relative_to(ROOT)}">\n{p.read_text()}\n</file>')
        # Give only the verdict lines of each workstream audit, to keep it cheap
        for audit in sorted(EXTRACTION.glob("*/AUDIT_*.md")):
            head = audit.read_text().split("## Issues")[0]
            work.append(f'<file path="{audit.relative_to(ROOT)}" excerpt="verdict">\n{head}\n</file>')
        task = ("Audit the ORCHESTRATOR: its decomposition of the work, its decisions, how it "
                "reconciles the workstreams, and whether its synthesis follows from the evidence. "
                "Brief coverage = whether the briefs themselves were well designed.")
        bundle, omitted = "\n\n".join(work), []
    else:
        target = (ROOT / target_arg).resolve()
        if not target.is_dir():
            sys.exit(f"Not a directory: {target}")
        out_dir = target
        bundle, omitted = collect(target)
        # Cross-workstream context: other workstreams' report summaries only
        others = []
        for rep in sorted(EXTRACTION.glob("*/REPORT.md")):
            if rep.parent != target:
                others.append(f'<summary path="{rep.relative_to(ROOT)}">\n{rep.read_text()[:3000]}\n</summary>')
        task = (f"Audit workstream `{target.name}` against its brief. "
                "Other workstreams' report openings are included only for consistency checks.")
        if others:
            bundle += "\n\n<other_workstreams>\n" + "\n".join(others) + "\n</other_workstreams>"

    omitted_txt = ("\n".join(f"- {o}" for o in omitted)) or "(none)"
    user = (f"<briefs>\n{briefs}\n</briefs>\n\n<work>\n{bundle}\n</work>\n\n"
            f"<not_included>\nFiles present but not sent (figures, large data):\n{omitted_txt}\n"
            f"</not_included>\n\n{task}")
    return user, out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="workstream directory, or 'orchestrator'")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(PRICES))
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--dry-run", action="store_true", help="count tokens only, no audit")
    args = ap.parse_args()

    user, out_dir = build_request(args.target)
    client = anthropic.Anthropic(api_key=load_key())
    messages = [{"role": "user", "content": user}]

    n_in = client.messages.count_tokens(model=args.model, system=SYSTEM, messages=messages).input_tokens
    p_in, p_out = PRICES[args.model]
    print(f"Input: {n_in} tokens (~${n_in * p_in / 1e6:.3f})  model={args.model}")
    if args.dry_run:
        return

    kwargs = dict(model=args.model, max_tokens=16000, system=SYSTEM, messages=messages)
    if args.model == "claude-haiku-4-5":
        response = client.messages.create(**kwargs)  # no effort/adaptive thinking on Haiku 4.5
    else:
        response = client.beta.messages.create(
            **kwargs,
            output_config={"effort": args.effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )

    if response.stop_reason == "refusal":
        sys.exit(f"Audit refused: {response.stop_details}")
    text = "".join(b.text for b in response.content if b.type == "text")
    if response.stop_reason == "max_tokens":
        text += "\n\n> WARNING: audit truncated at max_tokens."

    u = response.usage
    cost = u.input_tokens * p_in / 1e6 + u.output_tokens * p_out / 1e6
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = out_dir / f"AUDIT_{stamp}.md"
    header = (f"# External audit: {args.target}\n\n"
              f"_{stamp} · served by {response.model} · effort {args.effort} · "
              f"{u.input_tokens} in / {u.output_tokens} out · ~${cost:.3f}_\n\n")
    out.write_text(header + text)

    with USAGE_LOG.open("a") as f:
        f.write(json.dumps({"time": stamp, "target": args.target, "model": response.model,
                            "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                            "cost_usd": round(cost, 4), "request_id": response._request_id}) + "\n")
    print(f"Wrote {out.relative_to(ROOT)}  (~${cost:.3f})")


if __name__ == "__main__":
    main()
