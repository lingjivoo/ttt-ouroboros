"""Equivalence canaries for the admission-gate code paths.

Every new script in this project has shipped with a bug that only a run would
surface: a seed that was not idempotent, a dataset path that was a directory,
two inconsistent indexing schemes, a guard whose granularity did not match the
work it guarded. This round added five new options at once -- split_val, lam,
defer_content, defer_random, mixed -- so this checks each of them against a
setting where the answer is already known.

The checks are equivalences, not thresholds. Each one names a configuration
that must reduce EXACTLY to an existing arm, so a mismatch is a bug rather than
a judgement call:

    lam=1.0                       is a no-op
    defer_random  frac=0.0        commits nothing
    defer_random  frac=1.0        commits everything, i.e. defer_always
    defer_content thr=-inf        admits everything, i.e. defer_always
    defer_content thr=+inf        admits nothing
    no_split_val                  reports every probe, split_val reports half

Everything runs with --real-stream, so all paths are teacher-forced and
bitwise reproducible. That matters: 96.9% of generated tokens differ between
runs of the same seed, so a canary built on a sampled path can only ever
report noise. Forty chunks, which is the smallest horizon giving more than one
reported probe once the gate takes every other one -- these test control flow,
not effect.

Thresholds are passed as --content-thr=-1e9 rather than as two tokens: a bare
-1e9 is parsed as an option name, and argparse then reports the far less
obvious "expected one argument".

    python scripts/gate_canaries.py --ckpt <ckpt> --val <val.npy>
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def run(tmp, ckpt, val, name, extra, n_chunks, seed=42, real_stream=True):
    out = os.path.join(tmp, f"{name}.json")
    cmd = [sys.executable, os.path.join(HERE, "deferred.py"),
           "--ckpt", ckpt, "--val", val, "--seeds", str(seed),
           "--n-chunks", str(n_chunks), "--n-seqs", "2", "--book-offset", "2",
           "--out", out] + (["--real-stream"] if real_stream else []) + extra
    env = dict(os.environ, PYTHONPATH=ROOT)
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)
    if r.returncode != 0:
        raise RuntimeError(f"{name} failed:\n{r.stderr[-1500:]}")
    d = json.load(open(out))
    # _config is metadata, not a cell; every reader of these files has to skip it
    v = d[[k for k in d if not k.startswith("_")][0]]
    return {"probes": [(c, round(p, 6)) for c, p in v["probes"]],
            "committed": v.get("committed"), "held": v.get("held"),
            "admitted": v.get("admitted"),
            "probes_book": [(c, [round(x, 6) for x in xs])
                            for c, xs in (v.get("probes_book") or [])]}


def same(a, b):
    return a["probes"] == b["probes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val", required=True)
    # 40, not 16. Probes fall at (c-WARMUP) %% 8 == 7 and split_val gives every
    # other one to the gate, so 16 chunks contain a single probe, it is a
    # validation probe, and nothing is reported at all. 40 gives four probes:
    # two for the gate and two reported.
    ap.add_argument("--n-chunks", type=int, default=40)
    ap.add_argument("--keep", default="", help="directory to keep outputs in")
    args = ap.parse_args()

    tmp = args.keep or tempfile.mkdtemp(prefix="canary_")
    os.makedirs(tmp, exist_ok=True)
    # Start clean. deferred.py resumes per (mode, seed), so a rerun in a kept
    # directory returns the previous run's records and tests nothing -- which is
    # exactly how the probes_book check first "failed". The signature check in
    # deferred.py now catches this too; clearing here makes it impossible.
    stale = [f for f in os.listdir(tmp) if f.endswith(".json")]
    for f in stale:
        os.remove(os.path.join(tmp, f))
    if stale:
        print(f"cleared {len(stale)} stale result files from {tmp}", flush=True)
    N = args.n_chunks
    fails, checks = [], 0

    def check(name, cond, detail=""):
        nonlocal checks
        checks += 1
        print(f"[{'  ok  ' if cond else ' FAIL '}] {name}" + (f"  {detail}" if detail else ""),
              flush=True)
        if not cond:
            fails.append(name)

    print(f"running canaries at {N} chunks, teacher-forced, in {tmp}\n", flush=True)

    masked = run(tmp, args.ckpt, args.val, "masked", ["--mode", "masked"], N)
    always = run(tmp, args.ckpt, args.val, "always", ["--mode", "defer_always"], N)

    # 1. retention at 1.0 must be the identity
    lam1 = run(tmp, args.ckpt, args.val, "masked_lam1",
               ["--mode", "masked", "--lam", "1.0"], N)
    check("lam=1.0 is a no-op", same(masked, lam1),
          "" if same(masked, lam1) else "probes differ from the default path")

    # 2/3. the random gate collapses to its two endpoints
    r0 = run(tmp, args.ckpt, args.val, "rand0",
             ["--mode", "defer_random", "--commit-frac", "0.0"], N)
    check("defer_random frac=0 commits nothing", r0["committed"] == 0,
          f"committed {r0['committed']}/{r0['held']}")
    r1 = run(tmp, args.ckpt, args.val, "rand1",
             ["--mode", "defer_random", "--commit-frac", "1.0"], N)
    check("defer_random frac=1 commits everything", r1["committed"] == r1["held"],
          f"committed {r1['committed']}/{r1['held']}")
    check("defer_random frac=1 == defer_always", same(r1, always),
          "" if same(r1, always) else "committing everything took a different path")

    # 4/5. the content gate collapses to its two endpoints
    clo = run(tmp, args.ckpt, args.val, "content_lo",
              ["--mode", "defer_content", "--content-thr=-1e9"], N)
    check("content thr=-inf admits everything", clo["committed"] == clo["held"],
          f"committed {clo['committed']}/{clo['held']}")
    check("content thr=-inf == defer_always", same(clo, always),
          "" if same(clo, always) else "admitting everything took a different path")
    chi = run(tmp, args.ckpt, args.val, "content_hi",
              ["--mode", "defer_content", "--content-thr=1e9"], N)
    check("content thr=+inf admits nothing", chi["committed"] == 0,
          f"committed {chi['committed']}/{chi['held']}")

    # 6. the probe split -- the point of the correction
    split = run(tmp, args.ckpt, args.val, "dfu_split", ["--mode", "defer_update"], N)
    nosplit = run(tmp, args.ckpt, args.val, "dfu_nosplit",
                  ["--mode", "defer_update", "--no-split-val"], N)
    ns, ss = len(nosplit["probes"]), len(split["probes"])
    check("split_val reports about half the probes", ss * 2 in (ns, ns + 1),
          f"{ss} reported of {ns} total")
    idx = [c for c, _ in split["probes"]]
    allidx = [c for c, _ in nosplit["probes"]]
    every_other = allidx[1::2]
    check("reported probes are the ones the gate never saw", idx == every_other,
          f"reported {idx[:4]}... expected {every_other[:4]}...")

    # 8. forced admission at k=0 must change nothing
    seq = run(tmp, args.ckpt, args.val, "seq", ["--mode", "defer_seq"], N)
    seq0 = run(tmp, args.ckpt, args.val, "seq_f0",
               ["--mode", "defer_seq", "--force-admit", "0"], N)
    check("--force-admit 0 is a no-op", same(seq, seq0),
          "" if same(seq, seq0) else "probes differ from the unforced run")

    # 9/10. Forced admission has to be checked on a stream that HAS generated
    # updates. On the teacher-forced stream every chunk is real, so
    # forced_admissions selects from an empty list and both checks below pass
    # without testing anything -- which is exactly how they first read:
    # "of 0 held". The precondition is asserted so a vacuous pass fails.
    #
    # Generation is not bitwise reproducible, but admission COUNTS are the
    # count-type statistic this project has found does reproduce, and counts are
    # all these two checks read.
    gseq = run(tmp, args.ckpt, args.val, "gseq",
               ["--mode", "defer_seq"], N, real_stream=False)
    gheld = (gseq["admitted"] or {}).get("G_held", 0)
    check("the generated-stream cell actually holds generated updates",
          gheld > 0, f"G_held={gheld} (a zero here makes checks 10-11 vacuous)")

    gseqf = run(tmp, args.ckpt, args.val, "gseq_f3",
                ["--mode", "defer_seq", "--force-admit", "3"], N,
                real_stream=False)
    g_before = (gseq["admitted"] or {}).get("G", 0)
    g_after = (gseqf["admitted"] or {}).get("G", 0)
    want = min(3, gheld)
    check("forcing 3 admits at least that many generated updates",
          gheld > 0 and g_after >= want,
          f"generated admitted {g_before} -> {g_after}, at least {want} expected "
          f"of {(gseqf['admitted'] or {}).get('G_held', 0)} held")

    gupdf = run(tmp, args.ckpt, args.val, "gupd_f3",
                ["--mode", "defer_update", "--force-admit", "3"], N,
                real_stream=False)
    gu = (gupdf["admitted"] or {}).get("G", 0)
    check("both gate modes admit at least the forced count",
          gheld > 0 and gu >= want and g_after >= want,
          f"defer_update {gu}, defer_seq {g_after}, forced {want}")

    # 11. greedy forward selection can only accept what it has scored
    ok11 = (seq["committed"] or 0) <= (seq["held"] or 0)
    check("defer_seq commits no more than it held", ok11,
          f"committed {seq['committed']}/{seq['held']}")

    # 7. the per-book field must agree with the mean it replaced
    pbm = []
    for (c, m), (c2, xs) in zip(split["probes"], split.get("probes_book") or []):
        pbm.append(c == c2 and abs(sum(xs) / len(xs) - m) < 1e-6)
    check("per-book probes average to the reported probe",
          bool(pbm) and all(pbm),
          f"{sum(pbm)}/{len(pbm)} agree" if pbm else "probes_book missing")

    # 7. accounting closes
    a = split["admitted"] or {}
    tot = (a.get("R_held", 0) or 0) + (a.get("G_held", 0) or 0)
    check("held updates equal the R+G accounting", tot == split["held"],
          f"R_held+G_held={tot} vs held={split['held']}")

    print()
    if fails:
        print(f"{len(fails)} of {checks} FAILED: {', '.join(fails)}")
        print("These are equivalences, so a failure is a bug in the new options,")
        print("not a borderline result. Do not queue experiments on this build.")
    else:
        print(f"all {checks} canaries passed")
    return len(fails)


if __name__ == "__main__":
    sys.exit(main())
