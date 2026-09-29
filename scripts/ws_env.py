"""WebShop as a third agent environment, behind the same interface as sw_env.

Why WebShop. The two ScienceWorld tasks tried so far each satisfy one of the
two properties a write-authority testbed needs and not the other:
find-non-living-thing scores in {58, 67, 75} but never reaches a binary win, so
an outcome-verified policy writes nothing; inclined-plane-friction-named-surfaces
wins 10% of the time but scores only +-100, so its "graded" metric is binary with
a 12% base rate. WebShop's reward is continuous in [0, 1] -- the fraction of the
requested attributes, options and price the purchased item satisfies -- and its
goals are independent shopping instructions rather than variations of one task,
so it can have both resolution and a non-zero win rate at the same time.

No pyserini. Upstream searches through a prebuilt Lucene index, which needs Java
and an index build; the only thing the engine asks of that object is
`.search(query, k)` returning hits with `.docid`, and `.doc(docid)` returning
something whose `.raw()` is JSON carrying `id`. BM25Shim provides exactly that
over rank_bm25, which upstream already depends on and imports without using.
That keeps the install to six pure-Python packages and, more importantly, off
the pinned requirements.txt, which would downgrade torch to 1.11 and
transformers to 4.19 and break the model this project runs on.

Interface mirrors sw_env.SWBatchEnv: reset() -> (obs, admissible actions),
step(action) -> (obs, reward, done), and `last_score` for the graded outcome, so
agent_ttt's run_episode and the write policies work unchanged.
"""
import json
import os
import random
import re
import sys

WEBSHOP_DIR = os.environ.get("WEBSHOP_DIR", "")
# Catalogue built around the goal products (ws_build_catalogue.py). The shipped
# 10k subset was drawn at random from 1.18M products and intersects the
# goal-bearing ones almost nowhere, leaving 107 instructions; this one keeps all
# 10,136 of them plus distractors and yields 12,251.
WEBSHOP_DATA = os.environ.get("WEBSHOP_DATA", "")


class _Doc:
    def __init__(self, asin):
        self._raw = json.dumps({"id": asin})

    def raw(self):
        return self._raw


class _Hit:
    def __init__(self, docid):
        self.docid = docid


class BM25Shim:
    """Stands in for pyserini's LuceneSearcher over the product list.

    The engine only ever calls search() and doc(), so nothing else needs to
    exist. Ranking quality differs from Lucene's, which matters for comparing
    absolute scores against published WebShop numbers but not for comparing
    write policies against each other on this same searcher.
    """

    def __init__(self, products):
        from rank_bm25 import BM25Okapi
        self.asins = [p["asin"] for p in products]
        corpus = []
        for p in products:
            text = " ".join(str(p.get(k, "")) for k in
                            ("name", "Title", "small_description", "category",
                             "query", "product_category"))
            corpus.append(_tok(text))
        self.bm25 = BM25Okapi(corpus)

    def search(self, query, k=10):
        scores = self.bm25.get_scores(_tok(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [_Hit(self.asins[i]) for i in order]

    def doc(self, docid):
        return _Doc(docid)


def _tok(s):
    return re.findall(r"[a-z0-9]+", str(s).lower())


def _stub_pyserini():
    """Satisfy engine.py's module-level `from pyserini.search.lucene import
    LuceneSearcher` without installing pyserini.

    The import happens at module scope, so replacing init_search_engine
    afterwards is too late -- importing engine at all would fail first. Nothing
    ever calls this class: init_search_engine is replaced before any searcher is
    built, and it is the only place upstream constructs one. Installing pyserini
    instead would pull Java, faiss and a Lucene index build for a code path this
    wrapper does not use.
    """
    import types
    if "pyserini" in sys.modules:
        return
    for name in ("pyserini", "pyserini.search", "pyserini.search.lucene"):
        sys.modules[name] = types.ModuleType(name)

    class LuceneSearcher:  # noqa: N801 - mirrors the upstream name
        def __init__(self, *a, **k):
            raise RuntimeError(
                "the pyserini stub was actually constructed; "
                "init_search_engine should have been replaced first")

    sys.modules["pyserini.search.lucene"].LuceneSearcher = LuceneSearcher
    sys.modules["pyserini.search"].lucene = sys.modules["pyserini.search.lucene"]
    sys.modules["pyserini"].search = sys.modules["pyserini.search"]


class WSEnv:
    """One WebShop session, scored on the goal's reward.

    split='train'/'test' partitions the goals, not the products: the catalogue
    is shared, so a held-out goal is a held-out shopping instruction over items
    the agent may well have seen. That is the correct held-out unit here -- the
    thing being written into fast weights is how to shop, not which products
    exist -- but it is a weaker separation than ALFWorld's unseen layouts, and
    any claim built on it should say so.
    """

    def __init__(self, split="train", num_products=1000, max_steps=30,
                 max_actions=60, test_frac=0.2, seed=0):
        if not WEBSHOP_DIR or not WEBSHOP_DATA:
            raise RuntimeError(
                "set WEBSHOP_DIR to the upstream checkout and WEBSHOP_DATA "
                "to the prepared catalogue"
            )
        if WEBSHOP_DIR not in sys.path:
            sys.path.insert(0, WEBSHOP_DIR)
        _stub_pyserini()
        from web_agent_site.engine import engine as _eng

        # Inject the shim before anything constructs a searcher.
        prod_path = os.path.join(WEBSHOP_DATA, "items_shuffle.json")
        _eng.DEFAULT_FILE_PATH = prod_path
        _eng.DEFAULT_ATTR_PATH = os.path.join(WEBSHOP_DATA, "items_ins_v2.json")
        _eng.HUMAN_ATTR_PATH = os.path.join(WEBSHOP_DATA, "items_human_ins.json")
        self._searcher = None

        def _mk(num_products=None, _p=prod_path):
            # Build the index once: WebAgentTextEnv constructs a searcher per
            # instance, and BM25 over 50k products takes a while.
            if self._searcher is None:
                self._searcher = BM25Shim(_eng.load_products(
                    filepath=_p, num_products=None)[0])
            return self._searcher

        _eng.init_search_engine = _mk

        from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
        self.env = WebAgentTextEnv(observation_mode="text",
                                   file_path=prod_path,
                                   num_products=None,
                                   human_goals=True)
        self.max_steps = max_steps
        self.max_actions = max_actions
        n = len(self.env.server.goals) if hasattr(self.env, "server") else 0
        idx = list(range(n))
        random.Random(seed).shuffle(idx)
        cut = int(n * (1 - test_frac))
        self.goals = idx[:cut] if split == "train" else idx[cut:]
        assert self.goals, f"no {split} goals among {n}"
        self.order = list(range(len(self.goals)))
        self.i = 0
        self.last_score = 0.0

    def seed(self, n):
        # Rebuild the identity order before shuffling. Shuffling in place is not
        # idempotent, and two evaluations in one process would otherwise score
        # the same goals in a different order -- the defect that silently
        # misaligned per-episode pairing in the ScienceWorld harness.
        self.order = list(range(len(self.goals)))
        random.Random(n).shuffle(self.order)
        self.i = 0

    def _pack(self, obs, score):
        """The batched shape run_episode expects.

        run_episode is shared with ALFWorld and ScienceWorld and reads
        info["admissible_commands"][0] and info["won"][0]; presenting that shape
        here is what lets one episode runner drive three environments, which is
        the whole reason a difference between them cannot be the machinery. The
        first version of this wrapper returned (obs, actions) instead, so
        evaluation -- which has its own loop -- ran for thirty minutes and the
        training loop then died on its first episode, four times.
        """
        return {"admissible_commands": [self._actions()],
                "won": [bool(score >= 1.0)],
                "score": score}

    def reset(self):
        g = self.goals[self.order[self.i % len(self.order)]]
        self.i += 1
        obs, _ = self.env.reset(session=g)
        self.last_score = 0.0
        o = self._clip(obs)
        self.last_obs = o
        return [o], self._pack(o, 0.0)

    # Words that describe the product, not the framing. "Instruction:" is part
    # of the observation template and was going into the query as a search term.
    _STOP = {"instruction", "i", "am", "is", "a", "an", "the", "for", "of",
             "and", "or", "with", "that", "this", "my", "me", "need", "want",
             "looking", "find", "buy", "get", "please", "would", "like", "to",
             "in", "on", "at", "it", "its", "can", "you", "should", "be",
             "less", "lower", "than", "price", "dollars", "dollar"}

    def _queries(self):
        """One or two candidate queries built from the instruction.

        Two, not one: a long query is precise when the catalogue has the item
        and returns nothing useful when it does not, and a short one is the
        reverse. Giving the agent both leaves the choice to it, which is the
        only part of search it can express here.
        """
        words = [w for w in _tok(self.env.instruction_text)
                 if w not in self._STOP and not w.isdigit()]
        if not words:
            return []
        out = [" ".join(words[:8])]
        if len(words) > 4:
            short = " ".join(words[:4])
            if short != out[0]:
                out.append(short)
        return out

    def _actions(self):
        av = self.env.get_available_actions()
        acts = []
        if av.get("has_search_bar"):
            # A free-text search box is not an action set. The agent scores a
            # fixed list of candidates, so the query has to be constructed for
            # it; without this it can only click and never leaves the landing
            # page. That is a real limitation of this wrapper: the agent learns
            # what to click, not what to search for, so nothing here measures
            # query formulation. Policies are still comparable because every arm
            # gets the same candidate generator.
            acts += [f"search[{q}]" for q in self._queries()]
        acts += [f"click[{c}]" for c in av.get("clickables", [])]
        return acts[:self.max_actions]

    # A WebShop observation is a whole page -- a product listing runs well over
    # a thousand tokens -- and the shared prompt builder keeps the last eight
    # turns. Eight full pages reach twelve thousand tokens, and expanding that
    # KV cache once per candidate slice is what exhausted an 80 GiB card even
    # after the scoring was sliced. Truncate what goes into the history; the
    # CURRENT observation is passed in full, so nothing the agent needs to act
    # on right now is lost. Capping build_prompt instead would silently change
    # the ALFWorld and ScienceWorld runs, which share it.
    OBS_CAP = 600

    def _clip(self, obs):
        return obs if len(obs) <= self.OBS_CAP else obs[:self.OBS_CAP] + " ..."

    def step(self, actions):
        a = actions[0] if isinstance(actions, (list, tuple)) else actions
        obs, reward, done, _ = self.env.step(a)
        self.last_score = float(reward or 0.0)
        o = self._clip(obs)
        self.last_obs = o
        return [o], [self.last_score], [bool(done)], self._pack(o, self.last_score)
