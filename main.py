import argparse
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def filter_5core(df, user_col, item_col, min_cnt=5):
    while True:
        before = len(df)
        user_counts = df[user_col].value_counts()
        item_counts = df[item_col].value_counts()

        df = df[df[user_col].isin(user_counts[user_counts >= min_cnt].index)]
        df = df[df[item_col].isin(item_counts[item_counts >= min_cnt].index)]

        after = len(df)
        if after == before:
            break
    return df


def load_sequences(
    data_root,
    dataset="",
    user_col="reviewerID",
    item_col="asin",
    time_col="unixReviewTime",
    min_seq_len=3,
):
    data_root = Path(data_root)
    json_path = data_root / dataset
    if not json_path.exists():
        raise FileNotFoundError(f".json not found at: {json_path}")

    df = pd.read_json(json_path, lines=True)

    for col in [user_col, item_col, time_col]:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found in .json columns: {df.columns.tolist()}"
            )

    df = df[[user_col, item_col, time_col]].dropna()

    if df[time_col].dtype == object:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col])
        df[time_col] = (df[time_col].astype("int64") // 10**9).astype("int64")

    dataset_str = str(dataset).lower()
    if "beauty" not in dataset_str:
        df = filter_5core(df, user_col=user_col, item_col=item_col, min_cnt=5)

    df = df.sort_values([user_col, time_col])

    user_sequences = {}
    for u, g in df.groupby(user_col):
        seq = g[item_col].tolist()
        if len(seq) >= min_seq_len:
            user_sequences[u] = seq

    if not user_sequences:
        raise ValueError("No user sequences constructed (all shorter than min_seq_len).")

    return user_sequences


def build_item_mapping(user_sequences):
    item_set = set()
    for seq in user_sequences.values():
        item_set.update(seq)
    item2id = {item: idx + 1 for idx, item in enumerate(sorted(item_set))}
    id2item = {v: k for k, v in item2id.items()}
    return item2id, id2item


def build_luo_instances(user_sequences, item2id, max_len=50, alpha=1.0, inject_bias=True):
    train_seqs, train_tgts = [], []
    val_seqs, val_tgts = [], []
    test_seqs, test_tgts = [], []

    num_items = len(item2id)
    item_pop = [0.0] * (num_items + 1)

    mapped_seqs = []
    for seq in user_sequences.values():
        mapped = [item2id[i] for i in seq if i in item2id]
        if len(mapped) < 3:
            continue
        mapped_seqs.append(mapped)
        for iid in mapped:
            item_pop[iid] += 1.0

    if inject_bias:
        pop_pow = [c**alpha for c in item_pop]
        pop_pow[0] = 0.0
        Z = sum(pop_pow[1:])
        if Z > 0.0:
            item_prob = [c / Z for c in pop_pow]
        else:
            item_prob = [0.0] * (num_items + 1)
    else:
        item_prob = None

    for mapped in mapped_seqs:
        T = len(mapped)

        for t in range(1, T - 2):
            src = mapped[:t]
            tgt = mapped[t]
            if len(src) >= max_len:
                src = src[-max_len:]
            else:
                src = [0] * (max_len - len(src)) + src

            if inject_bias:
                p_keep = item_prob[tgt] if item_prob is not None else 1.0
                if p_keep <= 0.0:
                    continue
                if random.random() > p_keep:
                    continue

            train_seqs.append(src)
            train_tgts.append(tgt)

        val_src = mapped[: T - 2]
        val_tgt = mapped[T - 2]
        if len(val_src) >= max_len:
            val_src = val_src[-max_len:]
        else:
            val_src = [0] * (max_len - len(val_src)) + val_src
        val_seqs.append(val_src)
        val_tgts.append(val_tgt)

        test_src = mapped[: T - 1]
        test_tgt = mapped[T - 1]
        if len(test_src) >= max_len:
            test_src = test_src[-max_len:]
        else:
            test_src = [0] * (max_len - len(test_src)) + test_src
        test_seqs.append(test_src)
        test_tgts.append(test_tgt)

    return (train_seqs, train_tgts), (val_seqs, val_tgts), (test_seqs, test_tgts), item_pop


class SeqDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class SegmentAwareMixer(nn.Module):
    def __init__(self, num_channels: int, d: int, num_segments: int, rank: int = 4):
        super().__init__()
        self.num_channels = num_channels
        self.d = d

        self.seg_emb = nn.Embedding(num_segments, d)
        self.W_gamma = nn.Linear(d, rank, bias=True)
        self.U = nn.Parameter(torch.randn(num_channels, rank) * 0.02)
        self.V = nn.Parameter(torch.randn(num_channels, rank) * 0.02)

    def forward(self, X: torch.Tensor, seg_ids: torch.Tensor) -> torch.Tensor:
        N, V, d = X.shape
        assert V == self.num_channels and d == self.d

        e = self.seg_emb(seg_ids)               
        gamma = torch.sigmoid(self.W_gamma(e))  

        A = self.U.unsqueeze(0) * gamma.unsqueeze(1) 
        M = torch.matmul(A, self.V.t().unsqueeze(0)) 
        return torch.einsum("nvu,nud->nvd", M, X)    


class DGVDRec(nn.Module):
    def __init__(
        self,
        num_items: int,
        max_len: int = 50,
        d: int = 64,
        num_pop_bins: int = 10,
        item2popbin: Optional[torch.Tensor] = None,
        rank: int = 4,
        num_heads_var: int = 4,
        num_heads_time: int = 4,
        num_layers: int = 2,
        K: int = 20,
        lmax: int = 20,
        dropout: float = 0.1,
        delay_max: int = 0,
        delay_mode: str = "drop",       
    ):
        super().__init__()
        self.num_items = int(num_items)
        self.max_len = int(max_len)
        self.d = int(d)

        self.V = 3

        self.missing_id = self.num_items + 1
        self.num_tokens = self.num_items + 2

        self.item_emb = nn.Embedding(self.num_tokens, d, padding_idx=0)
        self.pos_emb = nn.Embedding(self.max_len, d)
        self.pop_emb = nn.Embedding(num_pop_bins + 1, d) 

        if item2popbin is None:
            item2popbin = torch.zeros(self.num_tokens, dtype=torch.long)
        else:
            if item2popbin.shape[0] != self.num_tokens:
                raise ValueError(
                    f"item2popbin must have shape ({self.num_tokens},), got {tuple(item2popbin.shape)}"
                )
        self.register_buffer("item2popbin", item2popbin)

        self.mixer = SegmentAwareMixer(
            num_channels=self.V,
            d=d,
            num_segments=num_pop_bins + 1,
            rank=rank,
        )

        self.var_attn = nn.MultiheadAttention(d, num_heads_var, dropout=dropout, batch_first=False)
        self.time_attn = nn.MultiheadAttention(d, num_heads_time, dropout=dropout, batch_first=False)

        self.fuse = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d, d),
        )

        self.lmax = int(lmax)
        self.K = int(min(K, self.max_len, self.lmax + 1))
        self.lag_emb = nn.Embedding(self.lmax + 1, d)

        self.Wi = nn.Linear(d, d, bias=False)
        self.Wj = nn.Linear(d, d, bias=False)
        self.Wtau = nn.Linear(d, d, bias=False)
        self.a = nn.Parameter(torch.randn(d) * 0.02)

        self.num_layers = int(num_layers)
        self.W_tau_layers = nn.ModuleList([nn.Embedding(self.lmax + 1, d * d) for _ in range(self.num_layers)])
        self.B_layers = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(self.num_layers)])

        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

        self.item_proj = nn.Linear(d, d, bias=False)
        self.scorer = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d, 1),
        )

        self.delay_max = int(delay_max)
        self.delay_mode = str(delay_mode)

    @staticmethod
    def _masked_mean(x: torch.Tensor, pad_mask: torch.Tensor, dim: int) -> torch.Tensor:
        keep = (~pad_mask).float()
        while keep.dim() < x.dim():
            keep = keep.unsqueeze(-1)
        x_sum = (x * keep).sum(dim=dim)
        denom = keep.sum(dim=dim).clamp_min(1.0)
        return x_sum / denom

    def _apply_delay(self, seq: torch.Tensor) -> torch.Tensor:
        if self.delay_max <= 0:
            return seq

        B, T = seq.shape
        device = seq.device

        lengths = (seq != 0).sum(dim=1)
        delays = torch.randint(0, self.delay_max + 1, (B,), device=device)
        delays = torch.minimum(delays, (lengths - 1).clamp_min(0)) 

        if self.delay_mode == "drop":
            out = seq.clone()
            for b in range(B):
                d = int(delays[b].item())
                if d <= 0:
                    continue
                l = int(lengths[b].item())
                keep = l - d
                start = T - l
                kept = out[b, start : start + keep]
                new = torch.zeros_like(out[b])
                new[-keep:] = kept
                out[b] = new
            return out

        if self.delay_mode == "mask_token":
            out = seq.clone()
            for b in range(B):
                d = int(delays[b].item())
                if d <= 0:
                    continue
                out[b, -d:] = self.missing_id
            return out

        raise ValueError(f"Unknown delay_mode: {self.delay_mode}")

    def compute_user(self, seq: torch.Tensor, return_routing: bool = False, apply_delay: bool = False):
        if apply_delay:
            seq = self._apply_delay(seq)

        B, T = seq.shape
        device = seq.device

        pad_mask = (seq == 0)

        pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        pos = self.pos_emb(pos_ids)
        item = self.item_emb(seq)
        popbin = self.item2popbin[seq]
        pop = self.pop_emb(popbin)

        X = torch.stack([item + pos, pos, pop + pos], dim=2)

        X_flat = X.reshape(B * T, self.V, self.d)
        seg_flat = popbin.reshape(B * T)
        X_mix = self.mixer(X_flat, seg_flat).reshape(B, T, self.V, self.d)

        R = []
        for v in range(self.V):
            R.append(self._masked_mean(X_mix[:, :, v, :], pad_mask, dim=1))
        R = torch.stack(R, dim=1)     
        R_t = R.permute(1, 0, 2)      

        Z, _ = self.var_attn(R_t, R_t, R_t, need_weights=False)
        Z = Z.permute(1, 0, 2)        
        z_pool = Z.mean(dim=1)        

        y = X_mix.mean(dim=2)         
        y_t = y.permute(1, 0, 2)      
        y_out, _ = self.time_attn(y_t, y_t, y_t, key_padding_mask=pad_mask, need_weights=False)
        y_out = y_out.permute(1, 0, 2)
        y_last = y_out[:, -1, :]      

        h0 = self.fuse(torch.cat([z_pool, y_last], dim=-1)) + y_last
        h0 = self.dropout(h0)

        src = y_out  

        K = self.K
        src_neigh = src[:, -K:, :]
        mask_neigh = pad_mask[:, -K:]
        src_neigh = torch.flip(src_neigh, dims=[1])
        mask_neigh = torch.flip(mask_neigh, dims=[1])

        tau_ids = torch.arange(0, K, device=device)   
        tau_emb = self.lag_emb(tau_ids).unsqueeze(0).expand(B, K, self.d)

        score = (self.Wi(h0).unsqueeze(1) + self.Wj(src_neigh) + self.Wtau(tau_emb)).tanh()
        score = (score * self.a.view(1, 1, -1)).sum(dim=-1)
        score = score.masked_fill(mask_neigh, float("-inf"))

        all_masked = torch.isinf(score).all(dim=1)
        if all_masked.any():
            score[all_masked, 0] = 0.0

        A = torch.softmax(score, dim=1)

        eps = 1e-12
        entropy = -(A * (A + eps).log()).sum(dim=1).mean()
        smooth = (
            ((A[:, 1:] - A[:, :-1]) ** 2).sum(dim=1).mean()
            if K > 1
            else torch.tensor(0.0, device=device)
        )

        h = h0
        for l in range(self.num_layers):
            W_tau = self.W_tau_layers[l](tau_ids).view(K, self.d, self.d)  
            msg_k = torch.einsum("kde,bke->bkd", W_tau, src_neigh)         
            msg = (A.unsqueeze(-1) * msg_k).sum(dim=1)                     
            h = self.act(msg + self.B_layers[l](h))
            h = self.dropout(h)

        if return_routing:
            return h, A, entropy, smooth
        return h

    def score_items(self, user_repr: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if item_ids.dim() == 1:
            item_ids = item_ids.unsqueeze(1)
            squeeze = True

        item_vec = self.item_proj(self.item_emb(item_ids))
        popbin = self.item2popbin[item_ids]
        item_vec = item_vec + self.pop_emb(popbin)

        B, N, _ = item_vec.shape
        u = user_repr.unsqueeze(1).expand(B, N, self.d)
        x = torch.cat([u, item_vec], dim=-1)
        s = self.scorer(x).squeeze(-1)
        return s.squeeze(1) if squeeze else s


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_pop_bins(item_pop: List[float], num_bins: int, num_items: int) -> torch.Tensor:
    counts = np.array(item_pop[1:], dtype=np.float64)
    order = np.argsort(counts)  
    item2bin = np.zeros(num_items + 2, dtype=np.int64)
    for rank, idx in enumerate(order):
        iid = idx + 1
        b = int(rank * num_bins / max(1, num_items)) + 1
        b = min(b, num_bins)
        item2bin[iid] = b
    item2bin[num_items + 1] = 0
    return torch.tensor(item2bin, dtype=torch.long)


def sample_negatives_train(pos_items: torch.Tensor, num_items: int, num_neg: int) -> torch.Tensor:
    B = pos_items.size(0)
    neg = torch.randint(1, num_items + 1, (B, num_neg), device=pos_items.device)
    for _ in range(10):
        mask = neg.eq(pos_items.unsqueeze(1))
        if not mask.any():
            break
        neg[mask] = torch.randint(1, num_items + 1, (mask.sum().item(),), device=pos_items.device)
    return neg


def sample_negatives_eval(seq_batch: torch.Tensor, pos_items: torch.Tensor, num_items: int, num_neg: int) -> torch.Tensor:
    B, _ = seq_batch.shape
    negs = torch.empty((B, num_neg), dtype=torch.long)

    seq_np = seq_batch.cpu().numpy()
    pos_np = pos_items.cpu().numpy()

    for b in range(B):
        seen = set(seq_np[b].tolist())
        seen.discard(0)
        seen.add(int(pos_np[b]))

        cand = []
        while len(cand) < num_neg:
            x = random.randint(1, num_items)
            if x not in seen:
                cand.append(x)
                seen.add(x)
        negs[b] = torch.tensor(cand, dtype=torch.long)

    return negs.to(seq_batch.device)


def metrics_from_scores(scores: torch.Tensor, k: int = 20) -> Tuple[float, float]:
    pos = scores[:, 0:1]
    rank = (scores[:, 1:] > pos).sum(dim=1) + 1
    hr = (rank <= k).float().mean().item()
    ndcg = ((rank <= k).float() / torch.log2(rank.float() + 1.0)).mean().item()
    return hr, ndcg


@torch.no_grad()
def evaluate(
    model: DGVDRec,
    loader: DataLoader,
    num_items: int,
    k_eval: int = 20,
    num_eval_neg: int = 100,
    device: str = "cpu",
    apply_delay: bool = False,
) -> Tuple[float, float]:
    model.eval()
    hrs, ndcgs = [], []
    for seq, tgt in loader:
        seq = seq.to(device)
        tgt = tgt.to(device)

        neg = sample_negatives_eval(seq, tgt, num_items, num_eval_neg)
        cands = torch.cat([tgt.unsqueeze(1), neg], dim=1)

        user, _, _, _ = model.compute_user(seq, return_routing=True, apply_delay=apply_delay)
        scores = model.score_items(user, cands)

        hr, ndcg = metrics_from_scores(scores, k=k_eval)
        hrs.append(hr)
        ndcgs.append(ndcg)

    return float(np.mean(hrs)), float(np.mean(ndcgs))


def train_model(
    data_root: str,
    dataset: str,
    user_col: str,
    item_col: str,
    time_col: str,
    max_len: int = 50,
    batch_size: int = 256,
    n_epochs: int = 200,
    patience: int = 50,
    k_eval: int = 20,
    alpha: float = 1.0,
    lr: float = 1e-3,
    lambda_l2: float = 1e-6,
    lambda_ent: float = 0.0,
    lambda_smooth: float = 0.0,
    # DGVD
    d: int = 64,
    num_layers: int = 2,
    num_heads_var: int = 4,
    num_heads_time: int = 4,
    K: int = 20,
    lmax: int = 20,
    num_neg: int = 50,
    num_eval_neg: int = 100,
    num_pop_bins: int = 10,
    rank: int = 4,
    dropout: float = 0.1,
    # delay simulation
    delay_max: int = 0,
    delay_mode: str = "drop",
    apply_delay_in_eval: bool = False,
    device: Optional[str] = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    user_sequences = load_sequences(
        data_root=data_root,
        dataset=dataset,
        user_col=user_col,
        item_col=item_col,
        time_col=time_col,
        min_seq_len=3,
    )
    item2id, id2item = build_item_mapping(user_sequences)

    (train_seqs, train_tgts), (val_seqs, val_tgts), (test_seqs, test_tgts), item_pop = build_luo_instances(
        user_sequences=user_sequences,
        item2id=item2id,
        max_len=max_len,
        alpha=alpha,
        inject_bias=True,
    )

    num_items = len(item2id)
    item2popbin = build_pop_bins(item_pop, num_bins=num_pop_bins, num_items=num_items)

    train_loader = DataLoader(SeqDataset(train_seqs, train_tgts), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SeqDataset(val_seqs, val_tgts), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(SeqDataset(test_seqs, test_tgts), batch_size=batch_size, shuffle=False)

    model = DGVDRec(
        num_items=num_items,
        max_len=max_len,
        d=d,
        num_pop_bins=num_pop_bins,
        item2popbin=item2popbin,
        rank=rank,
        num_heads_var=num_heads_var,
        num_heads_time=num_heads_time,
        num_layers=num_layers,
        K=K,
        lmax=lmax,
        dropout=dropout,
        delay_max=delay_max,
        delay_mode=delay_mode,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lambda_l2)

    best_val = -1.0
    best_state = None
    bad_epochs = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        losses = []

        for seq, tgt in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            seq = seq.to(device)
            tgt = tgt.to(device)

            neg = sample_negatives_train(tgt, num_items=num_items, num_neg=num_neg)
            cands = torch.cat([tgt.unsqueeze(1), neg], dim=1)

            user, _, ent, smooth = model.compute_user(
                seq,
                return_routing=True,
                apply_delay=(delay_max > 0),
            )
            scores = model.score_items(user, cands)
            labels = torch.zeros(scores.size(0), dtype=torch.long, device=device)

            loss_rank = F.cross_entropy(scores, labels)
            loss = loss_rank + lambda_ent * ent + lambda_smooth * smooth

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            losses.append(loss.item())

        val_hr, val_ndcg = evaluate(
            model,
            val_loader,
            num_items=num_items,
            k_eval=k_eval,
            num_eval_neg=num_eval_neg,
            device=device,
            apply_delay=(apply_delay_in_eval and delay_max > 0),
        )

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        print(f"[Epoch {epoch:03d}] loss={mean_loss:.4f}  val@{k_eval} HR={val_hr:.4f} NDCG={val_ndcg:.4f}")

        if val_ndcg > best_val:
            best_val = val_ndcg
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch} (best val NDCG={best_val:.4f}).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_hr, test_ndcg = evaluate(
        model,
        test_loader,
        num_items=num_items,
        k_eval=k_eval,
        num_eval_neg=num_eval_neg,
        device=device,
        apply_delay=(apply_delay_in_eval and delay_max > 0),
    )
    print(f"[Test@{k_eval}] HR={test_hr:.4f}  NDCG={test_ndcg:.4f}")

    return model, item2id, id2item


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--dataset", type=str, choices=["toys", "beauty", "sports", "yelp"], default="toys")
    p.add_argument("--max_len", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--n_epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--k_eval", type=int, default=20)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda_l2", type=float, default=1e-6)

    p.add_argument("--lambda_ent", type=float, default=0.0)
    p.add_argument("--lambda_smooth", type=float, default=0.0)

    p.add_argument("--d", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--num_heads_var", type=int, default=4)
    p.add_argument("--num_heads_time", type=int, default=4)
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--lmax", type=int, default=20)
    p.add_argument("--num_neg", type=int, default=50)
    p.add_argument("--num_eval_neg", type=int, default=100)
    p.add_argument("--num_pop_bins", type=int, default=10)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--delay_max", type=int, default=0)
    p.add_argument("--delay_mode", type=str, choices=["drop", "mask_token"], default="drop")
    p.add_argument("--apply_delay_in_eval", action="store_true")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    if args.dataset == "toys":
        data_root = "./datasets/"
        dataset_file = "Toys.json"
        user_col = "reviewerID"
        item_col = "asin"
        time_col = "unixReviewTime"
    elif args.dataset == "sports":
        data_root = "./datasets/"
        dataset_file = "Sports.json"
        user_col = "reviewerID"
        item_col = "asin"
        time_col = "unixReviewTime"
    elif args.dataset == "beauty":
        data_root = "./datasets/"
        dataset_file = "Beauty.json"
        user_col = "reviewerID"
        item_col = "asin"
        time_col = "unixReviewTime"
    elif args.dataset == "yelp":
        data_root = "./datasets/yelp"
        dataset_file = "yelp_academic_dataset_review.json"
        user_col = "user_id"
        item_col = "business_id"
        time_col = "date"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"Dataset = {args.dataset} ({dataset_file})")
    print(
        f"DGVD: d={args.d}, layers={args.num_layers}, K={args.K}, lmax={args.lmax}, "
        f"delay_max={args.delay_max}({args.delay_mode})"
    )

    train_model(
        data_root=data_root,
        dataset=dataset_file,
        user_col=user_col,
        item_col=item_col,
        time_col=time_col,
        max_len=args.max_len,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        patience=args.patience,
        k_eval=args.k_eval,
        alpha=args.alpha,
        lr=args.lr,
        lambda_l2=args.lambda_l2,
        lambda_ent=args.lambda_ent,
        lambda_smooth=args.lambda_smooth,
        d=args.d,
        num_layers=args.num_layers,
        num_heads_var=args.num_heads_var,
        num_heads_time=args.num_heads_time,
        K=args.K,
        lmax=args.lmax,
        num_neg=args.num_neg,
        num_eval_neg=args.num_eval_neg,
        num_pop_bins=args.num_pop_bins,
        rank=args.rank,
        dropout=args.dropout,
        delay_max=args.delay_max,
        delay_mode=args.delay_mode,
        apply_delay_in_eval=args.apply_delay_in_eval,
        device=args.device,
    )
