"""Frame embeddings from a pretrained image model, cached per recording in the media store."""

from __future__ import annotations

import time

import numpy as np
import torch

from .data import Recording, frames

MODEL, WEIGHTS = "MobileCLIP2-S0", "dfndr2b"
NAME = "mobileclip2-s0"
SIZE = 256


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load():
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(MODEL, pretrained=WEIGHTS)
    return model.eval().to(device())


def tensor(batch: list[np.ndarray]) -> torch.Tensor:
    # MobileCLIP2's preprocessing is a plain 0-1 scale (mean 0, std 1); only the centre crop is left out.
    x = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).float().div_(255.0)
    return x.to(device())


@torch.no_grad()
def embed(rec: Recording, model, fps: float = 1.0, batch: int = 64) -> None:
    out = rec.features(NAME)
    if out.exists():
        print(f"  {rec.id}: cached")
        return
    started = time.time()
    ts, xs, pending, pending_t = [], [], [], []

    def flush():
        e = model.encode_image(tensor(pending))
        e = torch.nn.functional.normalize(e, dim=-1)
        xs.append(e.cpu().numpy().astype(np.float16))
        ts.extend(pending_t)
        pending.clear()
        pending_t.clear()

    for t, img in frames(rec, fps, SIZE):
        pending.append(img)
        pending_t.append(t)
        if len(pending) == batch:
            flush()
    if pending:
        flush()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, t=np.array(ts, np.float32), x=np.concatenate(xs))
    print(f"  {rec.id}: {len(ts)} frames in {time.time() - started:.0f} s")
