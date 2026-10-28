"""
Two-tower retrieval.

The architectural bet is that a user and an item can be embedded *independently* and
compared by a dot product. That buys sublinear retrieval -- item vectors are indexed
offline, a query costs one ANN lookup -- and costs the ability to cross user and item
features before scoring, which is exactly what a factorization machine spends its
capacity on.

Two consequences shape how this is evaluated.

First, the headline metric is Recall@K at large K, not NDCG@10. A retrieval stage's job
is to get the relevant items into a shortlist that something else then ranks; scoring it
on top-10 ordering measures a job it was not built for.

Second, the latency argument cannot be demonstrated on this catalogue. With 3,883 items,
exhaustive scoring is milliseconds and an ANN index is pointless. The tower's advantage
is asymptotic and appears at 10^5+ items; that is stated rather than manufactured.

The user tower reads the same causal history window the FM's `history` block uses, so the
two models know the same things about a user at the same moment -- without it the
comparison would be against a weaker model than the FM's best configuration.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Recommender


def _sp_to_torch(M: sp.csr_matrix, device) -> torch.Tensor:
    M = M.tocoo()
    idx = torch.from_numpy(np.vstack([M.row, M.col])).long()
    val = torch.from_numpy(M.data.astype(np.float32))
    return torch.sparse_coo_tensor(idx, val, M.shape, device=device).coalesce()


class _UserTower(nn.Module):
    """User id + demographics + a mean-pooled bag of the causal history window."""

    def __init__(self, n_users, n_demo, n_items, emb=64, use_history=True, hidden=128):
        super().__init__()
        self.use_history = use_history
        self.id_emb = nn.Embedding(n_users, emb)
        self.demo = nn.Linear(n_demo, 32, bias=False)
        nn.init.normal_(self.id_emb.weight, 0, 0.05)
        d = emb + 32 + (emb if use_history else 0)
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, emb))

    def forward(self, uid, demo, hist_sp=None, item_emb=None):
        parts = [self.id_emb(uid), F.relu(self.demo(demo))]
        if self.use_history:
            # hist_sp is already 1/sqrt(n)-scaled, so this is a normalised bag-of-items
            parts.append(torch.sparse.mm(hist_sp, item_emb))
        return self.mlp(torch.cat(parts, dim=1))


class _ItemTower(nn.Module):
    def __init__(self, n_items, n_genres, n_years, emb=64, use_content=True, hidden=128):
        super().__init__()
        self.use_content = use_content
        self.id_emb = nn.Embedding(n_items, emb)
        nn.init.normal_(self.id_emb.weight, 0, 0.05)
        self.genre = nn.Linear(n_genres, 24, bias=False)
        self.year = nn.Linear(n_years, 12, bias=False)
        d = emb + (36 if use_content else 0)
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, emb))

    def forward(self, iid, genre, year):
        parts = [self.id_emb(iid)]
        if self.use_content:
            parts += [F.relu(self.genre(genre)), F.relu(self.year(year))]
        return self.mlp(torch.cat(parts, dim=1))


class TwoTower(Recommender):
    """Trained with a softmax over one positive and a set of negatives.

    `negatives` selects where those negatives come from, which is the one place the
    project's two inherited implementations disagreed:

      'in_batch'             the other positives in the minibatch. Standard for towers,
                             free, and implicitly popularity-weighted -- a frequent item
                             appears as a negative more often, which is a sampling bias
                             the logQ correction exists to undo.
      'unobserved'           uniform over items the user never interacted with.
      'unobserved_or_hated'  uniform over never-interacted items *and* items the user
                             rated below the positive threshold. Treats an explicit low
                             rating as a stronger negative than absence of evidence.
    """
    name = "two_tower"

    def __init__(self, emb: int = 64, lr: float = 3e-3, epochs: int = 12,
                 batch_size: int = 2048, n_negatives: int = 256,
                 negatives: str = "in_batch", temperature: float = 0.07,
                 use_history: bool = True, use_content: bool = True,
                 weight_decay: float = 1e-6, patience: int = 3, seed: int = 42,
                 logq: bool = True, device: str | None = None,
                 verbose: bool = True, name: str | None = None):
        self.emb, self.lr, self.epochs = emb, lr, epochs
        self.batch_size, self.n_negatives = batch_size, n_negatives
        self.negatives, self.temperature = negatives, temperature
        self.use_history, self.use_content = use_history, use_content
        self.weight_decay, self.patience, self.seed = weight_decay, patience, seed
        self.logq, self.verbose = logq, verbose
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if name: self.name = name
        self.negatives_note = negatives
        self.history_ = []

    # ---------------------------------------------------------------- setup

    def _static(self, ds):
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=self.device)
        self.genre_ = t(ds.item_genre.toarray())
        self.year_ = t(ds.item_year.toarray())
        self.demo_ = t(ds.user_demo.toarray())
        self.all_items_ = torch.arange(ds.n_items, device=self.device)
        # log-popularity for the logQ correction on in-batch negatives
        p = ds.item_train_count.astype(np.float64) + 1.0
        self.logq_ = t(np.log(p / p.sum()))

    def _item_matrix(self):
        return self.item_tower(self.all_items_, self.genre_, self.year_)

    # ------------------------------------------------------------------ fit

    def fit(self, ds, train_rows, val_rows=None):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self._static(ds)

        pos = train_rows[train_rows["label"] == 1].reset_index(drop=True)
        H = ds.history_matrix(pos) if self.use_history else None
        u = torch.as_tensor(pos["u"].to_numpy(np.int64), device=self.device)
        i = torch.as_tensor(pos["i"].to_numpy(np.int64), device=self.device)

        vpos = None
        if val_rows is not None:
            vpos = val_rows[val_rows["label"] == 1].reset_index(drop=True)
            Hv = ds.history_matrix(vpos) if self.use_history else None
            vu = torch.as_tensor(vpos["u"].to_numpy(np.int64), device=self.device)
            vi = torch.as_tensor(vpos["i"].to_numpy(np.int64), device=self.device)

        self.user_tower = _UserTower(ds.n_users, self.demo_.shape[1], ds.n_items,
                                     self.emb, self.use_history).to(self.device)
        self.item_tower = _ItemTower(ds.n_items, self.genre_.shape[1], self.year_.shape[1],
                                     self.emb, self.use_content).to(self.device)
        params = list(self.user_tower.parameters()) + list(self.item_tower.parameters())
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)

        # explicit-negative candidate pools, per policy
        if self.negatives != "in_batch":
            seen = {uu: set(ds._user_items[uu].tolist()) for uu in range(ds.n_users)}
            if self.negatives == "unobserved_or_hated":
                low = train_rows[train_rows["label"] == 0].groupby("u")["i"].apply(
                    lambda s: set(s.to_numpy().tolist())).to_dict()
            else:
                low = {}
            self._neg_pool = (seen, low)

        m = len(pos)
        best, best_state, bad = -np.inf, None, 0
        for ep in range(1, self.epochs + 1):
            self.user_tower.train(); self.item_tower.train()
            order = rng.permutation(m)
            tot, nb = 0.0, 0
            for s in range(0, m, self.batch_size):
                b = order[s: s + self.batch_size]
                if len(b) < 8:
                    continue
                loss = self._batch_loss(ds, pos, H, u, i, b, rng)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss.detach()); nb += 1
            rec = {"epoch": ep, "train_loss": tot / max(nb, 1)}
            if vpos is not None:
                rec["val_recall@100"] = self._quick_val(ds, vpos, Hv, vu, vi)
                if rec["val_recall@100"] > best + 1e-5:
                    best, bad = rec["val_recall@100"], 0
                    best_state = ({k: v.detach().clone() for k, v in self.user_tower.state_dict().items()},
                                  {k: v.detach().clone() for k, v in self.item_tower.state_dict().items()})
                else:
                    bad += 1
            self.history_.append(rec)
            if self.verbose:
                msg = f"    [{self.name}] epoch {ep:>2}  loss {rec['train_loss']:.4f}"
                if vpos is not None: msg += f"  val R@100 {rec['val_recall@100']:.4f}"
                print(msg, flush=True)
            if bad >= self.patience:
                if self.verbose: print(f"    [{self.name}] early stop at epoch {ep}", flush=True)
                break
        if best_state is not None:
            self.user_tower.load_state_dict(best_state[0])
            self.item_tower.load_state_dict(best_state[1])
        self._finalise()
        return self

    def _batch_loss(self, ds, pos, H, u, i, b, rng):
        ub, ib = u[b], i[b]
        item_all = self._item_matrix()
        hist = _sp_to_torch(H[b], self.device) if self.use_history else None
        ue = self.user_tower(ub, self.demo_[ub], hist, item_all)
        ue = F.normalize(ue, dim=1)

        if self.negatives == "in_batch":
            cand = ib                                   # every positive is another's negative
            ce = F.normalize(item_all[cand], dim=1)
            logits = ue @ ce.T / self.temperature
            if self.logq:
                logits = logits - self.logq_[cand][None, :]   # sampled-softmax correction
            target = torch.arange(len(b), device=self.device)
        else:
            seen, low = self._neg_pool
            negs = np.empty((len(b), self.n_negatives), np.int64)
            ub_np = ub.detach().cpu().numpy()
            for r, uu in enumerate(ub_np):
                forbidden = seen[int(uu)]
                allowed_extra = low.get(int(uu), set())
                drawn, guard = [], 0
                while len(drawn) < self.n_negatives and guard < 40:
                    c = rng.integers(0, ds.n_items, self.n_negatives)
                    for x in c:
                        x = int(x)
                        if x not in forbidden or x in allowed_extra:
                            drawn.append(x)
                            if len(drawn) == self.n_negatives: break
                    guard += 1
                while len(drawn) < self.n_negatives:
                    drawn.append(int(rng.integers(0, ds.n_items)))
                negs[r] = drawn
            negs_t = torch.as_tensor(negs, device=self.device)
            pe = F.normalize(item_all[ib], dim=1)
            ne = F.normalize(item_all[negs_t.reshape(-1)], dim=1).reshape(len(b), self.n_negatives, -1)
            pos_logit = (ue * pe).sum(1, keepdim=True) / self.temperature
            neg_logit = torch.einsum("bd,bnd->bn", ue, ne) / self.temperature
            logits = torch.cat([pos_logit, neg_logit], dim=1)
            target = torch.zeros(len(b), dtype=torch.long, device=self.device)
        return F.cross_entropy(logits, target)

    @torch.no_grad()
    def _quick_val(self, ds, vpos, Hv, vu, vi, k=100, sample=3000):
        """Recall@k of the true item among all items, on a sample of validation positives."""
        self.user_tower.eval(); self.item_tower.eval()
        n = min(sample, len(vpos))
        idx = np.arange(n)
        item_all = F.normalize(self._item_matrix(), dim=1)
        hist = _sp_to_torch(Hv[idx], self.device) if self.use_history else None
        ue = F.normalize(self.user_tower(vu[idx], self.demo_[vu[idx]], hist,
                                         self._item_matrix()), dim=1)
        scores = ue @ item_all.T
        topk = scores.topk(k, dim=1).indices
        return float((topk == vi[idx][:, None]).any(1).float().mean())

    @torch.no_grad()
    def _finalise(self):
        self.user_tower.eval(); self.item_tower.eval()
        self.item_emb_ = F.normalize(self._item_matrix(), dim=1)
        self._ucache = {}

    # -------------------------------------------------------------- scoring

    @torch.no_grad()
    def score(self, ds, u, items):
        u = int(u)
        if u not in self._ucache:
            frame = ds.scoring_frame(u, np.array([0], np.int32))
            hist = (_sp_to_torch(ds.history_matrix(frame), self.device)
                    if self.use_history else None)
            ut = torch.as_tensor([u], dtype=torch.long, device=self.device)
            e = self.user_tower(ut, self.demo_[ut], hist, self._item_matrix())
            self._ucache[u] = F.normalize(e, dim=1)
        it = torch.as_tensor(np.asarray(items), dtype=torch.long, device=self.device)
        return (self._ucache[u] @ self.item_emb_[it].T).squeeze(0).cpu().numpy()
