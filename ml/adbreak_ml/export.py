"""Export the model for Rust (`rust/crates/adbreak-model`): `model.json` and a test vector.

The exported model takes the inputs adbreak-model's `Row` has (picture embedding, caption timing, black and silence;
no caption text, which needs a tokenizer in Rust and is worth under a point of F1) and computes the table exactly
as `adbreak_model::Model::table` does. Training is as in `fusion.py`: the trees learn from out-of-fold frame
probabilities, and from each captioned recording a second time with its captions removed.
"""

from __future__ import annotations

import json
from pathlib import Path

import adbreak_rs as R
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from .data import BREAK, IGNORE, MINOR_KINDS, PROGRAMME, Recording, kind_labels
from .embed import NAME as EMBED, MODEL as IMAGE_MODEL
from .data import second_labels

from .fusion import WINDOWS, window_mean

RUST = "rust-v1"
FULL = "full-v1"


def embeddings(raw: np.ndarray, dim: int) -> np.ndarray:
    """The embedding columns of a features file; a second without a picture takes the previous second's (adbreak-model
    carries the previous probability)."""
    x = raw[:, :dim].copy()
    for i in range(len(x)):
        if np.isnan(x[i, 0]):
            x[i] = x[i - 1] if i else 0.0
    return x


class Data:
    """A recording's per-second inputs as `adbreak features` computes them (the code a recorder runs), and its labels.

    `views` are its picture embeddings from each version of the recording it has: the full-resolution recording's
    (`full-v1`, what a recorder scans) when ingest had it, and the small copy's (`rust-v1`, every recording). `x` is
    the first: the pictures nearest to what the model will see."""

    def __init__(self, rec: Recording, _name: str = RUST):
        self.rec = rec
        d = rec.features(RUST).parent
        meta = json.loads((d / f"{RUST}.json").read_text())
        raw = np.fromfile(d / f"{RUST}.f32", np.float32).reshape(meta["rows"], meta["dim"] + 6)
        self.views = [embeddings(raw, meta["dim"])]
        if (d / f"{FULL}.json").is_file():
            fm = json.loads((d / f"{FULL}.json").read_text())
            full = embeddings(np.fromfile(d / f"{FULL}.f32", np.float32).reshape(fm["rows"], fm["dim"] + 6), fm["dim"])
            # The same clock, but either may end a second or so earlier: cut or pad to the copy's rows.
            if len(full) < len(raw):
                full = np.concatenate([full, np.repeat(full[-1:], len(raw) - len(full), 0)])
            self.views.insert(0, full[: len(raw)])
        self.x = self.views[0]
        self.t = np.arange(meta["rows"], dtype=np.float64)
        self.y = second_labels(rec, self.t)
        self.ys = {k: kind_labels(rec, self.t, k) for k in MINOR_KINDS}
        self.cap = raw[:, meta["dim"] : meta["dim"] + 4].astype(np.float64)
        self.av = raw[:, meta["dim"] + 4 :].astype(np.float64)
        self.edges = np.array(meta["edges"]) if meta["edges"] else np.array([-1e9])
        from .text import cues, per_second

        vtt = d.parent / "captions.vtt"
        self.cues = cues(vtt.read_text(errors="ignore")) if vtt.is_file() else []
        self.tok_now, self.tok_trail = per_second(self.cues, len(self.t))
        # The text scores (out-of-fold in training), set by train_head / predict_head before table().
        self.tnow = self.ttrail = None

FORMAT = 1
TEXT_MAX = (30, 60)
EDGE_CAP = 120.0
# snap_black: a break's ends also snap to black pictures without a silence (a run of black seconds, at its middle):
# some channels open and close breaks with a black frame but no silence.
POST = {"threshold": 0.5, "min_len": 30.0, "max_gap": 15.0, "snap": 20.0, "snap_black": True}
# The other kinds come from the picture alone, averaged over ±KIND_SMOOTH s: with minutes of examples, trees on the
# other inputs learn the examples rather than the kind (leave-one-out AUC 0.64 against 0.82 for ad reads, 0.76
# against 0.94 for double boxes). Their runs are short (an ad read is often under 10 s): shorter runs kept, smaller
# gaps closed, nearer edges; thresholds from leave-one-out.
KIND_SMOOTH = 5
KIND_POST = {"double_box": {"threshold": 0.5, "min_len": 4.0, "max_gap": 3.0, "snap": 3.0},
             "ad_read": {"threshold": 0.4, "min_len": 4.0, "max_gap": 3.0, "snap": 3.0},
             "bumper": {"threshold": 0.5, "min_len": 2.0, "max_gap": 2.0, "snap": 3.0},
             "credits": {"threshold": 0.5, "min_len": 5.0, "max_gap": 3.0, "snap": 3.0}}


def target(d: "Data", kind: str) -> np.ndarray:
    """Per-second BREAK / PROGRAMME / IGNORE for an output: "ad" (commercial breaks) or one of MINOR_KINDS."""
    return d.y if kind == "ad" else d.ys[kind]


def feature_names() -> list[str]:
    return R.feature_names(WINDOWS, True)


def inputs(d: Data, drop_captions: bool = False):
    """Per-second (has, visible, roll_up, new, black, silent) and the edges, as adbreak-model's Row and Inputs."""
    cap = np.zeros_like(d.cap) if drop_captions else d.cap
    edges = [float(e) for e in d.edges if e > -1e8]
    return cap[:, 0], cap[:, 1], cap[:, 2], cap[:, 3], d.av[:, 0], d.av[:, 1], edges


def table(d: Data, p: np.ndarray, drop_captions: bool = False) -> np.ndarray:
    """The trees' inputs, by adbreak_model::table (through adbreak_rs): the code a recorder runs."""
    has, vis, roll, new, black, silent, edges = inputs(d, drop_captions)
    tnow, ttrail = getattr(d, "tnow", None), getattr(d, "ttrail", None)
    if tnow is None or drop_captions:
        tnow = ttrail = np.zeros(len(p))
    def f(a):
        return np.ascontiguousarray(a, np.float64)

    return R.table(f(p), f(has), f(vis), f(roll), f(new), f(black), f(silent), edges, WINDOWS, EDGE_CAP, f(tnow), f(ttrail))


class Never:
    """A frame model for a kind its training recordings have no examples of: probability 0."""

    coef_ = None

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.stack([np.ones(len(x)), np.zeros(len(x))], 1)


def frame_model(train: list[Data], kind: str = "ad"):
    """The picture model, on every version of each recording's pictures (each version weighted so a recording counts
    once): it learns the sharp pictures a recorder sees without the recordings that have them counting twice."""
    keep = [target(d, kind) != IGNORE for d in train]
    X = np.concatenate([v[k] for d, k in zip(train, keep) for v in d.views])
    Y = np.concatenate([target(d, kind)[k] == BREAK for d, k in zip(train, keep) for _ in d.views])
    W = np.concatenate([np.full(int(k.sum()), 1.0 / len(d.views)) for d, k in zip(train, keep) for _ in d.views])
    if not Y.any():
        return Never()
    # The other kinds are rare (minutes in hours): weighted so they count as much as everything else.
    return LogisticRegression(C=1.0, max_iter=3000, class_weight=None if kind == "ad" else "balanced").fit(X, Y, sample_weight=W)


def fit_trees(train: list[Data], oof: dict, kind: str = "ad") -> HistGradientBoostingClassifier:
    views = [(o, False) for o in train] + [(o, True) for o in train if o.cap[:, 0].any()]
    X = np.concatenate([table(o, oof[o.rec.id], drop)[target(o, kind) != IGNORE] for o, drop in views])
    Y = np.concatenate([target(o, kind)[target(o, kind) != IGNORE] == BREAK for o, _ in views])
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
                                          class_weight=None if kind == "ad" else "balanced", random_state=0).fit(X, Y)


def oof_text(train: list[Data]) -> None:
    """Out-of-fold text scores on each training recording (from text models trained on the others), for the trees."""
    from .text import TextHeads

    for o in train:
        o.tnow, o.ttrail = TextHeads([q for q in train if q is not o], BREAK, PROGRAMME).scores(o)


def train_head(train: list[Data], kind: str):
    """One output. Commercial breaks: a frame model and trees on its out-of-fold frame probabilities. The other
    kinds: a frame model alone (see KIND_SMOOTH)."""
    if kind != "ad":
        return frame_model(train, kind), None, None
    oof = {o.rec.id: frame_model([q for q in train if q is not o], kind).predict_proba(o.x)[:, 1] for o in train}
    oof_text(train)
    from .text import TextHeads

    return frame_model(train, kind), fit_trees(train, oof, kind), TextHeads(train, BREAK, PROGRAMME)


def predict_head(head, d: Data) -> tuple[np.ndarray, np.ndarray]:
    frame, trees, text = head
    if text is not None:
        d.tnow, d.ttrail = text.scores(d)
    p = frame.predict_proba(d.x)[:, 1]
    if trees is None:
        return p, window_mean(p, KIND_SMOOTH)
    return p, trees.predict_proba(table(d, p))[:, 1]


def has_kind(train: list[Data], kind: str) -> bool:
    return any((target(d, kind) == BREAK).any() for d in train)


def combine(ad: list, minor: dict, probs: dict | None = None) -> list[tuple[float, float, str]]:
    """Commercial breaks and the other kinds' runs as one list, by adbreak_model::combine (as Model::kind_breaks): a
    double box takes its time from a commercial break it overlaps; an ad read loses any part inside a commercial break
    or a double box; credits any part inside those or an ad read, and a bumper also inside credits. Pieces shorter
    than their kind's min_len are dropped, and commercial breaks shorter than POST's min_len."""
    runs = {k: ([tuple(b) for b in v], KIND_POST[k]["min_len"]) for k, v in minor.items()}
    return R.combine([tuple(b) for b in ad], runs, POST["min_len"])


def trees_json(clf: HistGradientBoostingClassifier) -> tuple[float, list]:
    out = []
    for it in clf._predictors:
        nodes = it[0].nodes
        out.append([
            {"leaf": True, "v": float(n["value"])} if n["is_leaf"] else
            {"f": int(n["feature_idx"]), "t": float(n["num_threshold"]), "l": int(n["left"]), "r": int(n["right"]),
             "missing_left": bool(n["missing_go_to_left"])}
            for n in nodes])
    return float(np.ravel(clf._baseline_prediction)[0]), out


def black_points(black: np.ndarray) -> list[float]:
    """adbreak_model::black_points: the middle of each run of seconds with any black."""
    return R.black_points(np.ascontiguousarray(black, np.float64))


def to_breaks(p: np.ndarray, edges: list[float], post: dict | None = None, black: np.ndarray | None = None) -> list[tuple[float, float]]:
    """adbreak_model::to_breaks (with the black points added when the settings say snap_black)."""
    POST = post or globals()["POST"]
    fallback = black_points(black) if POST.get("snap_black") and black is not None else []
    return R.to_breaks(np.ascontiguousarray(p, np.float64), [float(e) for e in edges], POST["threshold"], POST["min_len"],
                       POST["max_gap"], POST["snap"], fallback)


def model_version(text: str) -> str:
    """model.json's version (adbreak_model::version)."""
    return R.version(text)


def training_info(trees: HistGradientBoostingClassifier, labelled: list[Data], version: str) -> dict:
    """What the model was trained on and how training went, for the app's Model page (`model-info.json`)."""
    from datetime import datetime, timezone

    seconds = sum(int((d.y != IGNORE).sum()) for d in labelled)
    breaks = sum(int((d.y == BREAK).sum()) for d in labelled)
    # The trees stop early on a held-out tenth of their training seconds: its log loss each round, and the training's.
    loss = {}
    if getattr(trees, "validation_score_", None) is not None and len(trees.validation_score_):
        loss = {"train": [round(-float(v), 5) for v in trees.train_score_], "validation": [round(-float(v), 5) for v in trees.validation_score_]}
    return {
        "version": version,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trained_on": [d.rec.id for d in labelled],
        "seconds": seconds,
        "break_seconds": breaks,
        "kind_seconds": {k: int(sum((d.ys[k] == BREAK).sum() for d in labelled)) for k in MINOR_KINDS},
        "trees": int(trees.n_iter_),
        "max_trees": int(trees.max_iter),
        "loss": loss,
    }


def export(recs: list[Recording], out: Path, vector_secs: int = 600, note: str = "", name: str = "") -> None:
    from .data import holdout_ids

    data = [Data(r) for r in recs if r.features(RUST).with_suffix(".json").exists()]
    held_ids = holdout_ids()  # never trained on: they're what every model is scored on
    labelled = [d for d in data if (d.y != IGNORE).any() and d.rec.id not in held_ids]
    oof = {d.rec.id: frame_model([o for o in labelled if o is not d]).predict_proba(d.x)[:, 1] for d in labelled}
    oof_text(labelled)
    trees = fit_trees(labelled, oof)
    frame = frame_model(labelled)
    from . import text as T

    text_heads = T.export(T.TextHeads(labelled, BREAK, PROGRAMME))
    baseline, tj = trees_json(trees)
    model = {"format": FORMAT, "image_model": EMBED, "embedding_dim": int(labelled[0].x.shape[1]),
             "frame_coef": [float(c) for c in frame.coef_[0]], "frame_intercept": float(frame.intercept_[0]),
             "windows": WINDOWS, "edge_cap": EDGE_CAP, "features": feature_names(), "baseline": baseline, "trees": tj,
             "post": POST, "trained_on": [d.rec.id for d in labelled], "image_model_source": f"open_clip {IMAGE_MODEL}",
             "text": text_heads}
    # The other kinds' outputs (see KIND_SMOOTH), for those with examples to learn from.
    model["kinds"] = {}
    for k in MINOR_KINDS:
        f = frame_model(labelled, k)
        if f.coef_ is not None:
            model["kinds"][k] = {"frame_coef": [float(c) for c in f.coef_[0]], "frame_intercept": float(f.intercept_[0]),
                                 "smooth": KIND_SMOOTH, "post": KIND_POST[k]}
    text = json.dumps(model)
    from . import history as _history

    # A number to tell models apart by (model 1, 2, ...), and what changed, alongside the version hash.
    number = _history.next_number(out)
    # Its name: adbreak-0.<number> (or `name`, for a bigger change), then its size: the number of trees.
    info = {"number": number, "name": f"{name or f'adbreak-0.{number}'}-{int(trees.n_iter_)}", "note": note,
            **training_info(trees, labelled, model_version(text))}
    # A test vector: the first `vector_secs` seconds of a captioned recording, the exported model's outputs on it.
    d = next(d for d in labelled if d.cap[:, 0].any())
    n = min(vector_secs, len(d.t))
    x = d.x[:n].astype(np.float32)
    has, vis, roll, new, black, silent, edges = inputs(d)
    edges = [e for e in edges if e < n + EDGE_CAP]
    z = x.astype(np.float64) @ np.asarray(model["frame_coef"], np.float32).astype(np.float64) + model["frame_intercept"]
    p = 1 / (1 + np.exp(-z))
    sub = type("Sub", (), {})()
    sub.cap, sub.av, sub.edges = d.cap[:n], d.av[:n], np.array(edges) if edges else np.array([-1e9])
    # The captions of the vector's span, and the text scores from the exported weights, as a recorder computes them.
    vec_cues = [c for c in d.cues if c[0] < n]
    tok_now, tok_trail = T.per_second(vec_cues, n)
    zero = np.zeros(n)
    sub.tnow = T.score(tok_now, text_heads["now"]) if text_heads["now"] else zero
    sub.ttrail = T.score(tok_trail, text_heads["trail_head"]) if text_heads["trail_head"] else zero
    prob = trees.predict_proba(table(sub, p))[:, 1]

    def head_p(h: dict) -> np.ndarray:
        z = x.astype(np.float64) @ np.asarray(h["frame_coef"], np.float32).astype(np.float64) + h["frame_intercept"]
        return window_mean(1 / (1 + np.exp(-z)), h["smooth"])

    kp = {k: head_p(h) for k, h in model["kinds"].items()}
    kinds = combine(to_breaks(prob, edges, black=black[:n]), {k: to_breaks(v, edges, KIND_POST[k]) for k, v in kp.items()}, kp)
    vec = {"model": model_version(text), "recording": d.rec.id, "rows": [[x[i].tolist(), bool(has[i]), bool(vis[i]), bool(roll[i]), bool(new[i]),
                                             float(black[i]), float(silent[i])] for i in range(n)],
           "edges": edges, "captions": [list(c) for c in vec_cues], "probabilities": prob.tolist(), "breaks": to_breaks(prob, edges, black=black[:n]),
           "kind_probabilities": {k: v.tolist() for k, v in kp.items()}, "kind_breaks": [list(b) for b in kinds]}
    # Installed atomically, the test vector first: a recorder watching model.json reloads when it changes, and
    # checks the model against the test vector made for it (its "model" is model.json's version).
    out.mkdir(parents=True, exist_ok=True)
    from . import history

    history.keep(out)  # the model this replaces, so it can be put back
    (out / "scores.json").unlink(missing_ok=True)  # the old model's; the app writes the new one's
    for name, body in (("test_vector.json", json.dumps(vec)), ("model-info.json", json.dumps(info)), ("model.json", text)):
        tmp = out / f".{name}.tmp"
        tmp.write_text(body)
        tmp.replace(out / name)
    from . import runs

    runs.update(model_name=info["name"], model_version=info["version"], recordings=len(labelled))
    print(f"exported to {out}: {len(tj)} trees, trained on {len(labelled)} recordings; test vector {d.rec.id} "
          f"({n} s, {len(vec['breaks'])} breaks)")


def evaluate(recs: list[Recording], only: set[str] | None = None) -> None:
    """The shipped model's results for every recording, written as the "model" suggestion (score with `adbreak
    eval`) and, per second, as `features/model-out.json` (the picture score and the final probability, for the
    labelling page).

    With a held-out set (`adbreak holdout`): one model trained on the other labelled recordings gives every
    recording's results; only the held-out ones' are fair scores (the rest were in its training). Without one: each
    labelled recording leave-one-recording-out, from models trained on the others; unlabelled ones from models trained
    on every labelled recording (a first draft to label against)."""
    from .baseline import save_source
    from .data import labels_fingerprint

    data = [Data(r) for r in recs if r.features(RUST).with_suffix(".json").exists()]
    labelled = [d for d in data if (d.y != IGNORE).any()]
    fp = labels_fingerprint()
    # Models by the recordings they're trained on: with a held-out set, most results share one.
    trained: dict[tuple, tuple] = {}

    def heads(train: list[Data]):
        key = tuple(sorted(o.rec.id for o in train))
        if key not in trained:
            trained[key] = (train_head(train, "ad"), {k: train_head(train, k) for k in MINOR_KINDS if has_kind(train, k)})
        return trained[key]

    def write(d: Data, train: list[Data], how: str, source: str = "model") -> None:
        ad, kind_heads = heads(train)
        p_frame, prob = predict_head(ad, d)
        edges = [float(e) for e in d.edges if e > -1e8]
        probs = {k: predict_head(h, d)[1] for k, h in kind_heads.items()}
        minor = {k: to_breaks(p, edges, KIND_POST[k]) for k, p in probs.items()}
        found = combine(to_breaks(prob, edges, black=d.av[:, 0]), minor, probs)
        breaks = [{"start": a, "end": b, "kind": k} for a, b, k in found]
        save_source(d.rec, source, f"adbreak-model inputs (rust-v1), {how}", breaks)
        if source != "model":
            print(f"  {d.rec.id} ({source}): {sum(1 for *_, k in found if k == 'ad')} ad")
            return
        out = d.rec.features(RUST).parent / "model-out.json"
        metrics = {}
        m = d.y != IGNORE
        if m.any() and (d.y[m] == BREAK).any() and (d.y[m] != BREAK).any():
            from sklearn.metrics import log_loss, roc_auc_score

            y = d.y[m] == BREAK
            metrics = {"log_loss": round(float(log_loss(y, np.clip(prob[m], 1e-6, 1 - 1e-6))), 4), "auc": round(float(roc_auc_score(y, prob[m])), 4)}
        out.write_text(json.dumps({"trained_on": fp, "how": how, "metrics": metrics, "picture": [round(float(v), 3) for v in p_frame],
                                   "probability": [round(float(v), 3) for v in prob],
                                   "kinds": {k: [round(float(v), 3) for v in p] for k, p in probs.items()}}))
        n = {k: sum(1 for *_, kk in found if kk == k) for k in ("ad", *MINOR_KINDS)}
        print(f"  {d.rec.id}: " + ", ".join(f"{v} {k}" for k, v in n.items()))

    from . import runs

    from .data import holdout_ids

    held_ids = holdout_ids()
    held = [d for d in labelled if d.rec.id in held_ids]
    if held:
        train = [d for d in labelled if d.rec.id not in held_ids]
        jobs = [(d, train, f"trained on the {len(train)} training recordings, {'held out' if d in held else 'in its training'}", "model") for d in data]
        others_of = lambda d: [o for o in train if o.rec.network != d.rec.network]  # noqa: E731
        jobs += [(d, others_of(d), f"trained on the {len(others_of(d))} training recordings from other networks, held out", "model-new-network")
                 for d in held if others_of(d)]
    else:
        jobs = [(d, [o for o in labelled if o is not d], f"leave-one-recording-out over {len(labelled)}", "model") for d in labelled]
        jobs += [(d, labelled, f"trained on all {len(labelled)} labelled recordings", "model") for d in data if d not in labelled]
        for d in labelled:
            others = [o for o in labelled if o.rec.network != d.rec.network]
            if others:
                jobs.append((d, others, f"trained on the {len(others)} recordings from other networks", "model-new-network"))
    jobs = [j for j in jobs if only is None or j[0].rec.id in only]
    runs.update(recordings=len(labelled), labels=fp)
    for i, (d, train, how, source) in enumerate(jobs):
        runs.progress(f"evaluate {i + 1}/{len(jobs)}: {d.rec.id}{' (new network)' if source != 'model' else ''}")
        write(d, train, how, source)
