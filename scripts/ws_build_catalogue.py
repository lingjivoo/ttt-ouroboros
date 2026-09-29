"""Build a WebShop catalogue that actually contains the goal products.

WebShop ships a 10k product subset, and it yields 107 shopping instructions --
fewer independent evaluation units than the ScienceWorld task this environment
was supposed to improve on. The reason is not that the subset is small: the full
goal file names 10,136 products and the 10k subset was drawn at random from the
1.18M-product catalogue, so the two barely intersect. Sampling more products at
random would keep almost none of the goals.

So sample around the goals instead: keep every product a goal points at, then
pad with random distractors up to a target size. That gives all 12,251
instructions with a catalogue small enough for BM25.

Why not the full catalogue: rank_bm25 is pure Python and scores every document
on every query, so 1.18M products would make a 30-step episode take minutes.
Lucene is what upstream uses to avoid that, and avoiding Lucene -- and its Java
and index build -- is the reason this wrapper exists.

What this costs: absolute scores are no longer comparable with published WebShop
numbers, both because the catalogue differs and because BM25 ranks differently
from Lucene. Comparisons BETWEEN write policies on this catalogue are unaffected,
which is what the experiment needs; any absolute number should be labelled as
this catalogue's, not WebShop's.

    python scripts/ws_build_catalogue.py --src <dir> --out <dir> --distractors 40000
"""
import argparse
import json
import os
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory containing the full catalogue")
    ap.add_argument("--goals", required=True, help="upstream items_human_ins.json")
    ap.add_argument("--out", required=True, help="output directory for the filtered catalogue")
    ap.add_argument("--distractors", type=int, default=40000,
                    help="random non-goal products to pad with. They make search "
                         "non-trivial; too many and BM25 gets slow, too few and "
                         "every query returns the answer.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print("loading goals", flush=True)
    goals = json.load(open(args.goals))
    want = set(goals)
    print(f"  {len(want)} goal-bearing asins, "
          f"{sum(len(v) for v in goals.values())} instructions", flush=True)

    print("loading full catalogue (this is the 4 GB file)", flush=True)
    products = json.load(open(os.path.join(args.src, "items_shuffle.json")))
    print(f"  {len(products)} products", flush=True)

    keep, rest = [], []
    for p in products:
        (keep if p.get("asin") in want else rest).append(p)
    hit = {p["asin"] for p in keep}
    print(f"  goal products found: {len(keep)} of {len(want)} "
          f"({len(want - hit)} goals name a product not in the catalogue)",
          flush=True)

    random.Random(args.seed).shuffle(rest)
    keep += rest[:args.distractors]
    random.Random(args.seed + 1).shuffle(keep)
    print(f"  catalogue: {len(keep)} products", flush=True)

    out_products = os.path.join(args.out, "items_shuffle.json")
    json.dump(keep, open(out_products, "w"))

    # Attributes, restricted to the same asins so nothing dangles.
    attrs = json.load(open(os.path.join(args.src, "items_ins_v2.json")))
    asins = {p["asin"] for p in keep}
    sub = {a: v for a, v in attrs.items() if a in asins} if isinstance(attrs, dict) else attrs
    json.dump(sub, open(os.path.join(args.out, "items_ins_v2.json"), "w"))

    # Goals, restricted likewise: a goal whose product is absent can never be
    # satisfied and would sit in the evaluation as a guaranteed zero.
    gsub = {a: v for a, v in goals.items() if a in asins}
    json.dump(gsub, open(os.path.join(args.out, "items_human_ins.json"), "w"))

    n_ins = sum(len(v) for v in gsub.values())
    print(f"\nwrote {args.out}")
    print(f"  products     {len(keep)}")
    print(f"  attributes   {len(sub) if isinstance(sub, dict) else 'list'}")
    print(f"  goals        {len(gsub)} products, {n_ins} instructions")
    print(f"  a 80/20 split leaves about {int(n_ins * 0.2)} held-out instructions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
