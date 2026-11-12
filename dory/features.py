"""Text-baseline features from the NVIDIA prompt task/complexity classifier."""
import time

import numpy as np
import torch

from dory.nvidia_classifier import load_classifier

_KEYS = ["creativity_scope", "reasoning", "contextual_knowledge", "domain_knowledge",
         "constraint_ct", "number_of_few_shots", "prompt_complexity_score"]


def feature_rows(res: dict, prompts: list[str]):
    """Classifier output dict -> (X (n,9), is_code (n,))."""
    is_code = np.array([t == "Code Generation" for t in res["task_type_1"]])
    cols = [np.asarray(res[k], float) for k in _KEYS]
    cols += [np.log1p([len(p) for p in prompts]), is_code.astype(float)]
    return np.column_stack(cols), is_code


def text_features(prompts: list[str], batch_size: int = 16):
    """Returns (X, is_code, seconds); seconds = batch wall time / batch size, per prompt."""
    if not prompts:
        return np.empty((0, 9)), np.empty(0, bool), np.empty(0)
    model, tok, device = load_classifier()
    parts, secs = [], []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        t0 = time.perf_counter()
        enc = tok(chunk, return_tensors="pt", add_special_tokens=True, max_length=512,
                  padding=True, truncation=True).to(device)
        with torch.no_grad():
            res = model(enc)
        if device == "mps":
            torch.mps.synchronize()
        dt = (time.perf_counter() - t0) / len(chunk)
        parts.append(feature_rows(res, chunk))
        secs += [dt] * len(chunk)
    X = np.vstack([p[0] for p in parts])
    return X, np.concatenate([p[1] for p in parts]), np.array(secs)
