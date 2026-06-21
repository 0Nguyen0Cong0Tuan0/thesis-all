#!/usr/bin/env python3
"""
Verify what produced an output folder — reads train_info.json and reports the
self-identifying fields (run_tag, code_version, git_commit) and the key knobs,
so a mislabeled/stale/duplicate run is caught at a glance.

This exists because a folder named "spec_fastgs_output_v2.8" turned out to contain
v2.7 R3/R4 runs (renamed, not re-run). Folder names lie; train_info.json doesn't.

Usage:
  python tools/verify_run.py <dir>            # a model dir OR a parent of several
  python tools/verify_run.py <dir> --expect r5

Exit code is non-zero if --expect is given and any run's run_tag doesn't match.
"""
import os
import sys
import json
import glob
import argparse

KEYS = ["run_tag", "code_version", "git_commit", "spec_loss_weight", "spec_loss_mode",
        "normal_prior_weight", "normal_prior_start_iter", "final_gaussians",
        "training_time_formatted"]


def find_train_infos(path):
    if os.path.isfile(os.path.join(path, "train_info.json")):
        return [os.path.join(path, "train_info.json")]
    hits = set(glob.glob(os.path.join(path, "train_info.json")))
    hits |= set(glob.glob(os.path.join(path, "*", "train_info.json")))
    hits |= set(glob.glob(os.path.join(path, "**", "train_info.json"), recursive=True))
    return sorted(hits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="model dir or a parent containing several")
    ap.add_argument("--expect", default=None, help="expected run_tag (e.g. r5)")
    args = ap.parse_args()

    infos = find_train_infos(args.path)
    if not infos:
        raise SystemExit(f"No train_info.json under {args.path}")

    mismatches = 0
    for p in infos:
        info = json.load(open(p, encoding="utf-8"))
        print(f"\n=== {os.path.relpath(p, args.path)} ===")
        for k in KEYS:
            print(f"  {k:24}: {info.get(k, '(absent)')}")
        if args.expect is not None:
            tag = info.get("run_tag", "")
            ok = (tag == args.expect)
            print(f"  -> EXPECT run_tag='{args.expect}': {'MATCH' if ok else 'MISMATCH'}")
            if not ok:
                mismatches += 1

    if args.expect is not None and mismatches:
        raise SystemExit(f"\n{mismatches} run(s) did not match run_tag='{args.expect}'")


if __name__ == "__main__":
    main()
