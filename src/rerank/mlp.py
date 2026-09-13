"""Q2 Option B: a small neural ranker (MLP) over the same Q1 feature table.

Deliberately not NRMS here: NRMS reads the *articles themselves* (a token
sequence per news item) and is heavy enough - vocab, a news encoder, a user
encoder over raw click sequences - that it is Assignment 2's Q3 reproduced-
baseline deliverable in its own right, not the Q2 "small neural ranker" the
assignment explicitly also allows as "a simple MLP". This model instead scores
the same hand-crafted feature vector the GBDT sees, trained with a listwise
softmax loss so it is a genuine ranker rather than a pointwise classifier -
each impression's candidates compete against each other in one softmax, which
is what a `nDCG`-style evaluation actually rewards. Multi-click impressions
(27.9% of MIND) are handled by spreading the loss over every positive rather
than assuming exactly one.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.nn as nn

from src.eval import metrics as M


class MLPRanker(nn.Module):
    def __init__(self, n_features: int, hidden=(64, 32), dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class FeatureScaler:
    """Mean/std standardisation fit on train only, NaN -> train column mean,
    plus one missingness indicator per feature that has any NaN in train.

    A GBDT splits on NaN natively; a plain MLP cannot, and silently zeroing a
    NaN would tell the model "hours_since_last_click == 0" for a dataset where
    that feature is simply architecturally absent (MIND). The indicator column
    is what lets the network tell "zero" from "not available" apart.
    """

    def fit(self, X: np.ndarray) -> "FeatureScaler":
        self.has_nan_col = np.isnan(X).any(axis=0)
        # An architecturally-absent column (freshness on MIND, session
        # features on any dataset without has_session_id) is all-NaN, and
        # nanmean of an all-NaN slice warns by design - expected here, not a
        # bug, so it is silenced rather than left to alarm every run's stdout.
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            col_mean = np.nanmean(X, axis=0)
        col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)  # an all-NaN column
        self._col_mean = col_mean
        filled = np.where(np.isnan(X), col_mean, X)
        self.mean = filled.mean(axis=0)
        self.std = filled.std(axis=0)
        self.std = np.where(self.std < 1e-6, 1.0, self.std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        filled = np.where(np.isnan(X), self._col_mean, X)
        scaled = (filled - self.mean) / self.std
        missing = np.isnan(X)[:, self.has_nan_col].astype(np.float32)
        return np.concatenate([scaled, missing], axis=1).astype(np.float32)

    @property
    def n_out_features(self) -> int:
        return len(self.mean) + int(self.has_nan_col.sum())


def _forward_by_impression(model: MLPRanker, X_t: torch.Tensor, offsets: np.ndarray) -> torch.Tensor:
    """One forward pass over every candidate; caller slices by `offsets`."""
    return model(X_t)


def train(X_train: np.ndarray, y_train: np.ndarray, group_train: np.ndarray,
         X_val: np.ndarray, y_val: np.ndarray, group_val: np.ndarray,
         epochs: int = 30, batch_impressions: int = 64, lr: float = 1e-3,
         patience: int = 5, seed: int = 13, device: str | None = None):
    """Fit on train, early-stop on val AUC - the same split roles the GBDT uses."""
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    scaler = FeatureScaler().fit(X_train)
    Xtr = torch.from_numpy(scaler.transform(X_train)).to(device)
    ytr = torch.from_numpy(y_train.astype(np.float32)).to(device)
    Xva = torch.from_numpy(scaler.transform(X_val)).to(device)

    offsets_train = np.concatenate([[0], np.cumsum(group_train)])
    offsets_val = np.concatenate([[0], np.cumsum(group_val)])
    n_imp = len(group_train)

    model = MLPRanker(scaler.n_out_features).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)

    best_val_auc, best_state, bad_epochs = -1.0, None, 0

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(n_imp)
        for start in range(0, n_imp, batch_impressions):
            batch = order[start:start + batch_impressions]
            idx_ranges = [np.arange(offsets_train[i], offsets_train[i + 1]) for i in batch]
            idx = np.concatenate(idx_ranges) if idx_ranges else np.array([], dtype=np.int64)
            if len(idx) == 0:
                continue
            idx_t = torch.from_numpy(idx).to(device)
            scores = _forward_by_impression(model, Xtr[idx_t], offsets_train)

            loss = torch.zeros((), device=device)
            n_terms = 0
            cursor = 0
            for rng_i in idx_ranges:
                n = len(rng_i)
                if n == 0:
                    continue
                seg_scores = scores[cursor:cursor + n]
                seg_labels = ytr[torch.from_numpy(rng_i).to(device)]
                cursor += n
                pos = seg_labels.sum()
                if pos.item() == 0:
                    continue  # no positive to rank toward - mirrors AUC's "undefined" skip
                logp = torch.log_softmax(seg_scores, dim=0)
                loss = loss - (seg_labels * logp).sum() / pos
                n_terms += 1
            if n_terms == 0:
                continue
            loss = loss / n_terms
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_scores = model(Xva).cpu().numpy()
        val_per_imp = [val_scores[offsets_val[i]:offsets_val[i + 1]] for i in range(len(group_val))]
        val_labels_per_imp = [y_val[offsets_val[i]:offsets_val[i + 1]] for i in range(len(group_val))]
        aucs = [a for a in (M.auc(l, s) for l, s in zip(val_labels_per_imp, val_per_imp)) if a is not None]
        val_auc = float(np.mean(aucs)) if aucs else float("nan")

        if val_auc > best_val_auc:
            best_val_auc, bad_epochs = val_auc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, scaler, best_val_auc


def predict_per_impression(model: MLPRanker, scaler: FeatureScaler, X: np.ndarray,
                           offsets: np.ndarray, device: str | None = None) -> list[np.ndarray]:
    device = device or next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(scaler.transform(X)).to(device)
        flat = model(X_t).cpu().numpy()
    return [flat[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]
