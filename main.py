from __future__ import annotations

import argparse
import bisect
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def filter_5core(df: pd.DataFrame, user_col: str, item_col: str, min_cnt: int = 5) -> pd.DataFrame:
    """Iterative 5-core filtering."""
    while True:
        before = len(df)
        user_counts = df[user_col].value_counts()
        item_counts = df[item_col].value_counts()
        df = df[df[user_col].isin(user_counts[user_counts >= min_cnt].index)]
        df = df[df[item_col].isin(item_counts[item_counts >= min_cnt].index)]
        if len(df) == before:
            return df.reset_index(drop=True)


def load_raw_sequences(
    data_root: str,
    dataset: str,
    user_col: str,
    item_col: str,
    time_col: str,
    min_seq_len: int = 3,
    min_cnt: int = 5,
    truncate_to: Optional[int] = 200,
) -> Dict[str, List[Tuple[str, int]]]:
    """Load JSON-lines reviews and return raw per-user event-time sequences.

    Each sequence is a list of (raw_item_id, timestamp_seconds) sorted by event
    time.  Exact duplicate triples are removed to avoid artificially amplified
    graph edges.
    """
    path = Path(data_root) / dataset
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_json(path, lines=True)
    for col in [user_col, item_col, time_col]:
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not found. Available: {df.columns.tolist()}")

    df = df[[user_col, item_col, time_col]].dropna().drop_duplicates()
    if df[time_col].dtype == object:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col])
        df[time_col] = (df[time_col].astype("int64") // 10**9).astype("int64")
    else:
        df[time_col] = df[time_col].astype("int64")

    df = filter_5core(df, user_col=user_col, item_col=item_col, min_cnt=min_cnt)
    df = df.sort_values([user_col, time_col, item_col]).reset_index(drop=True)

    out: Dict[str, List[Tuple[str, int]]] = {}
    for u, g in df.groupby(user_col, sort=False):
        events = list(zip(g[item_col].astype(str).tolist(), g[time_col].astype(int).tolist()))
        if truncate_to is not None and len(events) > truncate_to:
            events = events[-truncate_to:]
        if len(events) >= min_seq_len:
            out[str(u)] = events
    if not out:
        raise ValueError("No valid user sequence after filtering.")
    return out


@dataclass
class MappedSequence:
    user_id: int
    items: List[int]
    timestamps: List[int]
    event_clocks: List[int]
    delays: List[int]
    obs_steps: List[int]
    user_delay_q: float


def build_mappings(raw_sequences: Dict[str, List[Tuple[str, int]]]) -> Tuple[Dict[str, int], Dict[str, int]]:
    user2id = {u: idx + 1 for idx, u in enumerate(sorted(raw_sequences.keys()))}
    item_set = sorted({item for seq in raw_sequences.values() for item, _ in seq})
    item2id = {item: idx + 1 for idx, item in enumerate(item_set)}
    return user2id, item2id


@dataclass
class DelayConfig:
    alpha: float = 1.0
    mu: float = 0.0
    sigma: float = 1.0
    lmax: int = 20
    variant: str = "stationary"
    mu_alpha: float = 0.0
    sigma_alpha: float = 0.35
    time_block_size: int = 20
    time_sigma: float = 0.35
    quantile_p: float = 0.95


def _sample_event_delays(
    T: int,
    rng: np.random.Generator,
    cfg: DelayConfig,
) -> Tuple[List[int], float]:
    """Sample clipped LogNormal integer delays for one user sequence.

    The delay is measured in the same clock units as the chosen experimental
    clock. With clock_mode="per_user", one unit means one event step in the
    user's own sequence, which matches Algorithm 2 and the severity table in
    the manuscript. With clock_mode="global", one unit means one event in the
    global timestamp-sorted stream.
    """
    if T <= 0:
        return [], 0.0

    user_scale = 1.0
    if cfg.variant in {"user", "user_time"}:
        user_scale = float(rng.lognormal(mean=cfg.mu_alpha, sigma=cfg.sigma_alpha))

    base = rng.lognormal(mean=cfg.mu, sigma=cfg.sigma, size=T)

    # For time-varying delay, keep one scale per coarse temporal block rather
    # than resampling the block factor for every event. This better matches the
    # manuscript description of transient congestion/device-state changes.
    block_scales: Dict[int, float] = {}
    if cfg.variant in {"time", "user_time"}:
        n_blocks = int(math.ceil(T / max(1, cfg.time_block_size)))
        for block_id in range(n_blocks):
            block_scales[block_id] = float(rng.lognormal(mean=0.0, sigma=cfg.time_sigma))

    delays: List[int] = []
    for idx in range(T):
        block_scale = 1.0
        if cfg.variant in {"time", "user_time"}:
            block_id = idx // max(1, cfg.time_block_size)
            block_scale = block_scales.get(block_id, 1.0)
        val = int(math.floor(cfg.alpha * user_scale * block_scale * float(base[idx])))
        delays.append(min(cfg.lmax, max(0, val)))
    q = float(np.quantile(delays, cfg.quantile_p)) if delays else 0.0
    return delays, q


def map_and_sample_delays(
    raw_sequences: Dict[str, List[Tuple[str, int]]],
    user2id: Dict[str, int],
    item2id: Dict[str, int],
    delay_cfg: DelayConfig,
    seed: int,
    clock_mode: str = "per_user",
) -> List[MappedSequence]:
    """Map raw sequences and sample observation delays.

    clock_mode="per_user" uses event_clocks = 1, ..., T_u inside each user's
    sequence. This exactly matches the manuscript's censored-history definition
    t_obs = t + Delta and makes the missing-ratio severity settings interpretable
    in units of user-event steps.

    clock_mode="global" keeps the timestamp-sorted global event clock. It is
    useful for production-style ablations, but its delay scale must be
    recalibrated because one delay unit then means one global log event rather
    than one user-history step.
    """
    if clock_mode not in {"per_user", "global"}:
        raise ValueError("clock_mode must be either 'per_user' or 'global'.")

    rng = np.random.default_rng(seed)

    clock_by_user_pos: Dict[Tuple[str, int], int] = {}
    if clock_mode == "global":
        global_events: List[Tuple[int, str, str, int]] = []
        for raw_u, seq in raw_sequences.items():
            for pos0, (raw_item, timestamp) in enumerate(seq):
                global_events.append((int(timestamp), str(raw_u), str(raw_item), int(pos0)))
        global_events.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
        clock_by_user_pos = {
            (raw_u, pos0): clock
            for clock, (_, raw_u, _, pos0) in enumerate(global_events, start=1)
        }

    mapped: List[MappedSequence] = []
    for raw_u, seq in raw_sequences.items():
        uid = user2id[raw_u]
        items: List[int] = []
        ts: List[int] = []
        event_clocks: List[int] = []
        for pos0, (item, timestamp) in enumerate(seq):
            if item not in item2id:
                continue
            items.append(item2id[item])
            ts.append(int(timestamp))
            if clock_mode == "per_user":
                event_clocks.append(pos0 + 1)
            else:
                event_clocks.append(clock_by_user_pos[(str(raw_u), pos0)])

        if len(items) < 3:
            continue
        delays, user_q = _sample_event_delays(len(items), rng, delay_cfg)
        obs_steps = [event_clocks[j] + delays[j] for j in range(len(items))]
        mapped.append(MappedSequence(uid, items, ts, event_clocks, delays, obs_steps, user_q))
    if not mapped:
        raise ValueError("No mapped sequence with length >= 3.")
    return mapped


def build_item_pop(mapped_sequences: Sequence[MappedSequence], num_items: int) -> List[float]:
    pop = [0.0] * (num_items + 1)
    for seq in mapped_sequences:
        for iid in seq.items:
            pop[iid] += 1.0
    return pop


def build_pop_bins(item_pop: Sequence[float], num_bins: int, num_items: int) -> torch.Tensor:
    counts = np.asarray(item_pop[1:], dtype=np.float64)
    order = np.argsort(counts)
    item2bin = np.zeros(num_items + 2, dtype=np.int64)
    for rank, idx in enumerate(order):
        iid = idx + 1
        b = int(rank * num_bins / max(1, num_items)) + 1
        item2bin[iid] = min(b, num_bins)
    return torch.tensor(item2bin, dtype=torch.long)


@dataclass(frozen=True)
class BufferRecord:
    obs_step: int
    event_clock: int
    event_step: int
    user_id: int
    item_id: int
    delay: int


class ObservedBipartiteGraph:
    """Bounded user/item buffers for the observation-time bipartite graph.

    The buffers are pre-built for offline experiments, but the insert_event
    method mirrors the online event-driven update path described in the paper.
    Retrieval uses the target decision clock c. In the default per-user
    experimental clock, c is the target user's event step; in global mode, c is
    the timestamp-sorted global event clock.
    """

    def __init__(self, num_users: int, num_items: int, k_max: int, ttl: Optional[int] = None):
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.k_max = int(k_max)
        self.ttl = ttl
        self.user_buffers: Dict[int, List[BufferRecord]] = {u: [] for u in range(1, num_users + 1)}
        self.item_buffers: Dict[int, List[BufferRecord]] = {i: [] for i in range(1, num_items + 1)}
        self._user_obs_index: Dict[int, List[int]] = {}
        self._item_obs_index: Dict[int, List[int]] = {}

    def insert_event(self, rec: BufferRecord) -> None:
        self.user_buffers.setdefault(rec.user_id, []).append(rec)
        self.item_buffers.setdefault(rec.item_id, []).append(rec)

    def finalize(self) -> None:
        for uid, buf in self.user_buffers.items():
            buf.sort(key=lambda r: (r.obs_step, r.event_step, r.item_id))
            self._user_obs_index[uid] = [r.obs_step for r in buf]
        for iid, buf in self.item_buffers.items():
            buf.sort(key=lambda r: (r.obs_step, r.event_step, r.user_id))
            self._item_obs_index[iid] = [r.obs_step for r in buf]

    @classmethod
    def from_sequences(
        cls,
        mapped_sequences: Sequence[MappedSequence],
        num_users: int,
        num_items: int,
        k_max: int,
        ttl: Optional[int] = None,
    ) -> "ObservedBipartiteGraph":
        graph = cls(num_users=num_users, num_items=num_items, k_max=k_max, ttl=ttl)
        for seq in mapped_sequences:
            for pos, (item_id, event_clock, delay, obs_step) in enumerate(zip(seq.items, seq.event_clocks, seq.delays, seq.obs_steps), start=1):
                graph.insert_event(BufferRecord(obs_step=obs_step, event_clock=event_clock, event_step=pos, user_id=seq.user_id, item_id=item_id, delay=delay))
        graph.finalize()
        return graph

    def _visible_prefix(self, buf: List[BufferRecord], obs_index: List[int], c: int) -> List[BufferRecord]:
        end = bisect.bisect_right(obs_index, c)
        recs = buf[:end]
        if self.ttl is not None:
            recs = [r for r in recs if c - r.obs_step <= self.ttl]
        return recs

    def get_user_history(self, user_id: int, c: int, target_event_step: int, max_len: int) -> List[BufferRecord]:
        buf = self.user_buffers.get(int(user_id), [])
        idx = self._user_obs_index.get(int(user_id), [])
        recs = self._visible_prefix(buf, idx, int(c))

        recs = [r for r in recs if r.event_step < target_event_step and r.event_clock < c]
        recs.sort(key=lambda r: r.event_step)
        return recs[-max_len:]

    def get_user_degree(self, user_id: int, c: int, target_event_step: Optional[int] = None) -> int:
        recs = self._visible_prefix(self.user_buffers.get(int(user_id), []), self._user_obs_index.get(int(user_id), []), int(c))
        recs = [r for r in recs if r.event_clock < c]
        if target_event_step is not None:
            recs = [r for r in recs if r.event_step < target_event_step]
        return len(recs)

    def get_item_neighbor_users(
        self,
        item_id: int,
        c: int,
        k_max: int,
        exclude_user: Optional[int] = None,
        exclude_event_step_ge: Optional[int] = None,
    ) -> List[BufferRecord]:
        buf = self.item_buffers.get(int(item_id), [])
        idx = self._item_obs_index.get(int(item_id), [])
        recs = self._visible_prefix(buf, idx, int(c))
        recs = [r for r in recs if r.event_clock < c]
        if exclude_user is not None and exclude_event_step_ge is not None:
            recs = [r for r in recs if not (r.user_id == exclude_user and r.event_step >= exclude_event_step_ge)]

        recs.sort(key=lambda r: (r.obs_step, r.event_clock), reverse=True)
        return recs[:k_max]

    def get_item_degree(self, item_id: int, c: int) -> int:
        recs = self._visible_prefix(self.item_buffers.get(int(item_id), []), self._item_obs_index.get(int(item_id), []), int(c))
        return len([r for r in recs if r.event_clock < c])

    def user_delay_quantile(self, user_id: int, c: int, p: float, fallback: float, target_event_step: Optional[int] = None) -> float:
        recs = self._visible_prefix(self.user_buffers.get(int(user_id), []), self._user_obs_index.get(int(user_id), []), int(c))
        recs = [r for r in recs if r.event_clock < c]
        if target_event_step is not None:
            recs = [r for r in recs if r.event_step < target_event_step]
        if not recs:
            return float(fallback)
        return float(np.quantile([r.delay for r in recs], p))


@dataclass
class CensoredInstance:
    user_id: int
    target_item: int
    target_step: int
    target_clock: int
    seq_items: List[int]
    seq_lags: List[int]
    seq_obs_ages: List[int]
    observed_len: int
    missing_ratio: float
    delay_q: float


class CensoredSeqDataset(Dataset):
    def __init__(
        self,
        instances: Sequence[CensoredInstance],
        max_len: int,
        num_items: int,
        seen_by_user: Dict[int, set],
        num_eval_neg: int = 0,
        seed: int = 42,
    ):
        self.instances = list(instances)
        self.max_len = int(max_len)
        self.num_items = int(num_items)
        self.seen_by_user = seen_by_user
        self.num_eval_neg = int(num_eval_neg)
        self.eval_negs: Optional[torch.Tensor] = None
        if self.num_eval_neg > 0:
            self.eval_negs = self._precompute_eval_negs(seed)

    def _pad(self, values: List[int], pad_value: int = 0) -> List[int]:
        values = values[-self.max_len:]
        return [pad_value] * (self.max_len - len(values)) + values

    def _precompute_eval_negs(self, seed: int) -> torch.Tensor:
        rng = random.Random(seed)
        negs = torch.zeros((len(self.instances), self.num_eval_neg), dtype=torch.long)
        all_items = set(range(1, self.num_items + 1))
        for idx, inst in enumerate(self.instances):
            seen = set(self.seen_by_user.get(inst.user_id, set()))
            seen.add(inst.target_item)
            pool = list(all_items - seen)
            if not pool:

                pool = [x for x in range(1, self.num_items + 1) if x != inst.target_item]
            if not pool:
                pool = [inst.target_item]
            vals = [rng.choice(pool) for _ in range(self.num_eval_neg)]
            negs[idx] = torch.tensor(vals, dtype=torch.long)
        return negs

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int):
        inst = self.instances[idx]
        seq_items = torch.tensor(self._pad(inst.seq_items, 0), dtype=torch.long)
        seq_lags = torch.tensor(self._pad(inst.seq_lags, 0), dtype=torch.long)
        seq_obs_age = torch.tensor(self._pad(inst.seq_obs_ages, 0), dtype=torch.long)
        eval_negs = (
            self.eval_negs[idx]
            if self.eval_negs is not None
            else torch.empty(0, dtype=torch.long)
        )
        return {
            "index": torch.tensor(idx, dtype=torch.long),
            "user_id": torch.tensor(inst.user_id, dtype=torch.long),
            "seq_items": seq_items,
            "seq_lags": seq_lags,
            "seq_obs_age": seq_obs_age,
            "target": torch.tensor(inst.target_item, dtype=torch.long),
            "target_step": torch.tensor(inst.target_step, dtype=torch.long),
            "target_clock": torch.tensor(inst.target_clock, dtype=torch.long),
            "observed_len": torch.tensor(inst.observed_len, dtype=torch.long),
            "missing_ratio": torch.tensor(inst.missing_ratio, dtype=torch.float32),
            "delay_q": torch.tensor(inst.delay_q, dtype=torch.float32),
            "eval_negs": eval_negs,
        }


def build_instances(
    mapped_sequences: Sequence[MappedSequence],
    graph: ObservedBipartiteGraph,
    split: str,
    max_len: int,
    delay_quantile_p: float,
) -> List[CensoredInstance]:
    instances: List[CensoredInstance] = []
    global_delays = [d for seq in mapped_sequences for d in seq.delays]
    global_q = float(np.quantile(global_delays, delay_quantile_p)) if global_delays else 0.0

    for seq in mapped_sequences:
        T = len(seq.items)
        if T < 3:
            continue
        if split == "train":
            target_positions = range(1, T - 2)
        elif split == "val":
            target_positions = [T - 2]
        elif split == "test":
            target_positions = [T - 1]
        else:
            raise ValueError(f"Unknown split: {split}")

        for pos0 in target_positions:
            target_step = pos0 + 1
            target_clock = seq.event_clocks[pos0]
            target_item = seq.items[pos0]
            visible = graph.get_user_history(seq.user_id, c=target_clock, target_event_step=target_step, max_len=max_len)
            seq_items = [r.item_id for r in visible]
            seq_lags = [max(0, target_step - r.event_step) for r in visible]
            seq_obs_ages = [max(0, target_clock - r.obs_step) for r in visible]
            obs_len = len(visible)
            denom = max(1, target_step - 1)
            missing_ratio = 1.0 - float(obs_len) / float(denom)
            delay_q = graph.user_delay_quantile(seq.user_id, target_clock, p=delay_quantile_p, fallback=global_q, target_event_step=target_step)
            instances.append(
                CensoredInstance(
                    user_id=seq.user_id,
                    target_item=target_item,
                    target_step=target_step,
                    target_clock=target_clock,
                    seq_items=seq_items,
                    seq_lags=seq_lags,
                    seq_obs_ages=seq_obs_ages,
                    observed_len=obs_len,
                    missing_ratio=missing_ratio,
                    delay_q=delay_q,
                )
            )
    return instances


def build_seen_by_user(mapped_sequences: Sequence[MappedSequence]) -> Dict[int, set]:
    return {seq.user_id: set(seq.items) for seq in mapped_sequences}


class SegmentAwareMixer(nn.Module):
    def __init__(self, num_channels: int, d: int, num_segments: int, rank: int = 4):
        super().__init__()
        self.num_channels = int(num_channels)
        self.d = int(d)
        self.seg_emb = nn.Embedding(num_segments, d)
        self.W_gamma = nn.Linear(d, rank, bias=True)
        self.U = nn.Parameter(torch.randn(num_channels, rank) * 0.02)
        self.V = nn.Parameter(torch.randn(num_channels, rank) * 0.02)

    def forward(self, X: torch.Tensor, seg_ids: torch.Tensor) -> torch.Tensor:
        N, V, d = X.shape
        if V != self.num_channels or d != self.d:
            raise ValueError(f"Expected X shape (*,{self.num_channels},{self.d}), got {tuple(X.shape)}")
        e = self.seg_emb(seg_ids.clamp_min(0))
        gamma = torch.sigmoid(self.W_gamma(e))
        A = self.U.unsqueeze(0) * gamma.unsqueeze(1)
        M = torch.matmul(A, self.V.t().unsqueeze(0))
        return torch.einsum("nvu,nud->nvd", M, X)


class DGVDRec(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        max_len: int = 50,
        d: int = 128,
        num_pop_bins: int = 10,
        item2popbin: Optional[torch.Tensor] = None,
        rank: int = 4,
        num_heads_var: int = 4,
        num_heads_time: int = 4,
        num_layers: int = 2,
        k_min: int = 5,
        k_max: int = 20,
        l_min: int = 5,
        lmax: int = 20,
        beta_k: float = 4.0,
        beta_l: float = 0.5,
        adaptive_k: bool = True,
        adaptive_l: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.max_len = int(max_len)
        self.d = int(d)
        self.num_channels = 3

        self.k_min = int(k_min)
        self.k_max = int(min(k_max, max_len))
        self.l_min = int(l_min)
        self.lmax = int(lmax)
        self.beta_k = float(beta_k)
        self.beta_l = float(beta_l)
        self.adaptive_k = bool(adaptive_k)
        self.adaptive_l = bool(adaptive_l)

        self.num_tokens = self.num_items + 2
        self.user_emb = nn.Embedding(self.num_users + 1, d, padding_idx=0)
        self.item_emb = nn.Embedding(self.num_tokens, d, padding_idx=0)
        self.pos_emb = nn.Embedding(self.max_len, d)
        self.pop_emb = nn.Embedding(num_pop_bins + 1, d)
        self.lag_emb = nn.Embedding(self.lmax + 1, d)
        self.age_emb = nn.Embedding(self.lmax + 1, d)

        if item2popbin is None:
            item2popbin = torch.zeros(self.num_tokens, dtype=torch.long)
        if item2popbin.shape[0] != self.num_tokens:
            raise ValueError(f"item2popbin must have shape ({self.num_tokens},), got {tuple(item2popbin.shape)}")
        self.register_buffer("item2popbin", item2popbin)

        self.mixer = SegmentAwareMixer(self.num_channels, d, num_pop_bins + 1, rank=rank)
        self.temporal_context = nn.Linear(d, d)
        self.var_attn = nn.MultiheadAttention(d, num_heads_var, dropout=dropout, batch_first=False)
        self.time_attn = nn.MultiheadAttention(d, num_heads_time, dropout=dropout, batch_first=False)
        self.fuse_gate = nn.Linear(2 * d, d)
        self.fuse_mlp = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d))


        self.Wi = nn.Linear(d, d, bias=False)
        self.Wj = nn.Linear(d, d, bias=False)
        self.Wtau = nn.Linear(d, d, bias=False)
        self.Wedge = nn.Linear(3, d, bias=False)
        self.a = nn.Parameter(torch.randn(d) * 0.02)
        self.W_tau_layers = nn.ModuleList([nn.Embedding(self.lmax + 1, d * d) for _ in range(num_layers)])
        self.B_layers = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(num_layers)])


        self.item_query = nn.Linear(d, d, bias=False)
        self.item_user_key = nn.Linear(d, d, bias=False)
        self.item_lag_key = nn.Linear(d, d, bias=False)
        self.item_neighbor_proj = nn.Linear(d, d, bias=False)
        self.item_a = nn.Parameter(torch.randn(d) * 0.02)


        self.item_proj = nn.Linear(d, d, bias=False)
        self.user_proj = nn.Linear(d, d, bias=False)
        self.bilinear = nn.Linear(d, d, bias=False)
        self.pair_mlp = nn.Sequential(nn.Linear(5 * d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 1))
        self.item_bias = nn.Embedding(self.num_tokens, 1)

        self.num_layers = int(num_layers)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    @staticmethod
    def _masked_mean(x: torch.Tensor, pad_mask: torch.Tensor, dim: int) -> torch.Tensor:
        keep = (~pad_mask).float()
        while keep.dim() < x.dim():
            keep = keep.unsqueeze(-1)
        denom = keep.sum(dim=dim).clamp_min(1.0)
        return (x * keep).sum(dim=dim) / denom

    def _active_k_from_degree(self, degree: torch.Tensor) -> torch.Tensor:
        if not self.adaptive_k:
            return torch.full_like(degree, self.k_max)
        vals = torch.ceil(self.beta_k * torch.sqrt(degree.float() + 1.0)).long()
        vals = torch.clamp(vals, min=self.k_min, max=self.k_max)
        return vals

    def _active_l_from_delay(self, missing_ratio: torch.Tensor, delay_q: torch.Tensor) -> torch.Tensor:
        if not self.adaptive_l:
            return torch.full_like(missing_ratio.long(), self.lmax)
        vals = torch.ceil(delay_q.float() + self.beta_l * missing_ratio.float() * float(self.lmax)).long()
        vals = torch.clamp(vals, min=self.l_min, max=self.lmax)
        return vals

    def compute_user(
        self,
        user_ids: torch.Tensor,
        seq_items: torch.Tensor,
        seq_lags: torch.Tensor,
        seq_obs_age: torch.Tensor,
        observed_len: torch.Tensor,
        missing_ratio: torch.Tensor,
        delay_q: torch.Tensor,
        return_routing: bool = False,
    ):
        B, T = seq_items.shape
        device = seq_items.device
        pad_mask = seq_items.eq(0)
        safe_pad_mask = pad_mask.clone()
        all_pad = safe_pad_mask.all(dim=1)
        if all_pad.any():
            safe_pad_mask[all_pad, -1] = False

        pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        pos = self.pos_emb(pos_ids)
        item = self.item_emb(seq_items)
        popbin = self.item2popbin[seq_items]
        pop = self.pop_emb(popbin)
        lag_ids_full = seq_lags.clamp(0, self.lmax)
        age_ids_full = seq_obs_age.clamp(0, self.lmax)


        X = torch.stack([
            item + pos,
            self.lag_emb(lag_ids_full) + self.age_emb(age_ids_full),
            pop + pos,
        ], dim=2)

        X_flat = X.reshape(B * T, self.num_channels, self.d)
        seg_flat = popbin.reshape(B * T)
        X_mix = self.mixer(X_flat, seg_flat).reshape(B, T, self.num_channels, self.d)


        y = X_mix.mean(dim=2)
        y_t = y.permute(1, 0, 2)
        y_out, _ = self.time_attn(y_t, y_t, y_t, key_padding_mask=safe_pad_mask, need_weights=False)
        y_out = y_out.permute(1, 0, 2)
        y_last = y_out[:, -1, :]

        R = []
        temporal_ctx = self.temporal_context(y_last)
        for v in range(self.num_channels):
            token = self._masked_mean(X_mix[:, :, v, :], pad_mask, dim=1)
            R.append(token + temporal_ctx)
        R = torch.stack(R, dim=1)
        Z, _ = self.var_attn(R.permute(1, 0, 2), R.permute(1, 0, 2), R.permute(1, 0, 2), need_weights=False)
        Z = Z.permute(1, 0, 2)
        z_pool = Z.mean(dim=1)

        gate = torch.sigmoid(self.fuse_gate(torch.cat([z_pool, y_last], dim=-1)))
        h0 = self.fuse_mlp(torch.cat([gate * z_pool, (1.0 - gate) * y_last], dim=-1)) + y_last
        h0 = h0 + self.user_emb(user_ids)
        h0 = self.dropout(h0)

        active_k = self._active_k_from_degree(observed_len.to(device))
        active_l = self._active_l_from_delay(missing_ratio.to(device), delay_q.to(device))

        K = self.k_max
        src = y_out
        src_neigh = torch.flip(src[:, -K:, :], dims=[1])
        mask_neigh = torch.flip(pad_mask[:, -K:], dims=[1])
        lag_neigh = torch.flip(seq_lags[:, -K:], dims=[1]).clamp(0, self.lmax)
        age_neigh = torch.flip(seq_obs_age[:, -K:], dims=[1]).clamp(0, self.lmax)

        Ltau = self.lmax + 1
        rank_ids = torch.arange(K, device=device).view(1, K, 1)
        tau_ids = torch.arange(Ltau, device=device).view(1, 1, Ltau)

        budget_mask = rank_ids >= active_k.view(B, 1, 1)
        lag_budget_mask = tau_ids > active_l.view(B, 1, 1)
        final_mask = mask_neigh.unsqueeze(-1) | budget_mask | lag_budget_mask
        all_masked = final_mask.view(B, -1).all(dim=1)
        if all_masked.any():

            final_mask = final_mask.clone()
            final_mask[all_masked, 0, 0] = False
            src_neigh = src_neigh.clone()
            src_neigh[all_masked, 0, :] = 0.0
            lag_neigh = lag_neigh.clone()
            age_neigh = age_neigh.clone()
            lag_neigh[all_masked, 0] = 0
            age_neigh[all_masked, 0] = 0
        src_neigh = src_neigh.masked_fill(mask_neigh.unsqueeze(-1), 0.0)

        tau_emb = self.lag_emb(tau_ids.squeeze(0).squeeze(0)).view(1, 1, Ltau, self.d)
        edge_feat = torch.stack([
            lag_neigh.float() / max(1.0, float(self.lmax)),
            age_neigh.float() / max(1.0, float(self.lmax)),
            missing_ratio.to(device).unsqueeze(1).expand(B, K),
        ], dim=-1).unsqueeze(2).expand(B, K, Ltau, 3)

        score = (
            self.Wi(h0).view(B, 1, 1, self.d)
            + self.Wj(src_neigh).view(B, K, 1, self.d)
            + self.Wtau(tau_emb)
            + self.Wedge(edge_feat)
        ).tanh()
        score = (score * self.a.view(1, 1, 1, -1)).sum(dim=-1)
        score = score.masked_fill(final_mask, float("-inf"))

        A = torch.softmax(score.view(B, -1), dim=1).view(B, K, Ltau)
        A = A.masked_fill(final_mask, 0.0)
        A = A / A.sum(dim=(1, 2), keepdim=True).clamp_min(1e-12)

        eps = 1e-12
        entropy = -(A * (A + eps).log()).sum(dim=(1, 2)).mean()
        lag_marginal = A.sum(dim=1)
        smooth = ((lag_marginal[:, 1:] - lag_marginal[:, :-1]) ** 2).sum(dim=1).mean() if Ltau > 1 else torch.tensor(0.0, device=device)

        h = h0
        tau_vector = torch.arange(Ltau, device=device)
        for layer_idx in range(self.num_layers):
            W_tau = self.W_tau_layers[layer_idx](tau_vector).view(Ltau, self.d, self.d)
            msg_grid = torch.einsum("lde,bke->bkld", W_tau, src_neigh)
            msg = (A.unsqueeze(-1) * msg_grid).sum(dim=(1, 2))
            h = self.act(msg + self.B_layers[layer_idx](h))
            h = self.dropout(h)

        lag_summary = torch.matmul(lag_marginal, self.lag_emb.weight)
        stats = {
            "active_k": active_k.float().mean().detach(),
            "active_l": active_l.float().mean().detach(),

            "active_candidates": (active_k.float() * (active_l.float() + 1.0)).mean().detach(),

            "active_observed_pairs": (~final_mask).float().sum(dim=1).mean().detach(),
        }
        if return_routing:
            return h, lag_summary, A, entropy, smooth, stats
        return h, lag_summary

    def _encode_candidate_items(
        self,
        item_ids: torch.Tensor,
        item_neighbor_users: torch.Tensor,
        item_neighbor_lags: torch.Tensor,
        item_active_k: torch.Tensor,
        active_l: torch.Tensor,
    ) -> torch.Tensor:
        B, N = item_ids.shape
        K = item_neighbor_users.size(-1)
        item_vec = self.item_proj(self.item_emb(item_ids)) + self.pop_emb(self.item2popbin[item_ids])

        neigh_users = item_neighbor_users.clamp_min(0)
        neigh_vec = self.user_emb(neigh_users)
        lag_ids = item_neighbor_lags.clamp(0, self.lmax)
        lag_vec = self.lag_emb(lag_ids)

        rank_ids = torch.arange(K, device=item_ids.device).view(1, 1, K)
        mask = neigh_users.eq(0)
        mask = mask | (rank_ids >= item_active_k.unsqueeze(-1))
        mask = mask | (lag_ids > active_l.view(B, 1, 1))
        neigh_vec = neigh_vec.masked_fill(neigh_users.eq(0).unsqueeze(-1), 0.0)

        q = self.item_query(item_vec).unsqueeze(2)
        score = (q + self.item_user_key(neigh_vec) + self.item_lag_key(lag_vec)).tanh()
        score = (score * self.item_a.view(1, 1, 1, -1)).sum(dim=-1)
        all_masked = mask.all(dim=-1)
        if all_masked.any():
            mask = mask.clone()
            mask[all_masked, 0] = False
        score = score.masked_fill(mask, float("-inf"))
        W = torch.softmax(score, dim=-1).masked_fill(mask, 0.0)
        W = W / W.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        agg = (W.unsqueeze(-1) * neigh_vec).sum(dim=2)
        return item_vec + self.item_neighbor_proj(agg)

    def score_items(
        self,
        user_repr: torch.Tensor,
        lag_summary: torch.Tensor,
        item_ids: torch.Tensor,
        item_neighbor_users: torch.Tensor,
        item_neighbor_lags: torch.Tensor,
        item_active_k: torch.Tensor,
        missing_ratio: torch.Tensor,
        delay_q: torch.Tensor,
    ) -> torch.Tensor:
        squeeze = False
        if item_ids.dim() == 1:
            item_ids = item_ids.unsqueeze(1)
            item_neighbor_users = item_neighbor_users.unsqueeze(1)
            item_neighbor_lags = item_neighbor_lags.unsqueeze(1)
            item_active_k = item_active_k.unsqueeze(1)
            squeeze = True

        active_l = self._active_l_from_delay(missing_ratio.to(item_ids.device), delay_q.to(item_ids.device))
        item_repr = self._encode_candidate_items(item_ids, item_neighbor_users, item_neighbor_lags, item_active_k, active_l)
        item_repr = self.item_proj(item_repr)
        user_repr = self.user_proj(user_repr)

        B, N, _ = item_repr.shape
        u = user_repr.unsqueeze(1).expand(B, N, self.d)
        r = lag_summary.unsqueeze(1).expand(B, N, self.d)
        bilinear = (self.bilinear(u) * item_repr).sum(dim=-1)
        pair = torch.cat([u, item_repr, u * item_repr, torch.abs(u - item_repr), r], dim=-1)
        mlp_score = self.pair_mlp(pair).squeeze(-1)
        bias = self.item_bias(item_ids).squeeze(-1)
        score = bilinear + mlp_score + bias
        return score.squeeze(1) if squeeze else score


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_negatives_train(
    user_ids: torch.Tensor,
    pos_items: torch.Tensor,
    seen_by_user: Dict[int, set],
    num_items: int,
    num_neg: int,
    device: torch.device,
) -> torch.Tensor:
    out = torch.empty((pos_items.size(0), num_neg), dtype=torch.long, device=device)
    all_items = set(range(1, num_items + 1))
    for b in range(pos_items.size(0)):
        uid = int(user_ids[b].item())
        pos = int(pos_items[b].item())
        seen = set(seen_by_user.get(uid, set()))
        seen.add(pos)
        pool = list(all_items - seen)
        if not pool:

            pool = [x for x in range(1, num_items + 1) if x != pos]
        if not pool:
            pool = [pos]
        vals = [random.choice(pool) for _ in range(num_neg)]
        out[b] = torch.tensor(vals, dtype=torch.long, device=device)
    return out


def metrics_from_scores(scores: torch.Tensor, k: int = 20) -> Tuple[float, float]:
    pos = scores[:, 0:1]
    rank = (scores[:, 1:] > pos).sum(dim=1) + 1
    recall = (rank <= k).float().mean().item()
    ndcg = ((rank <= k).float() / torch.log2(rank.float() + 1.0)).mean().item()
    return recall, ndcg


def build_item_neighbor_tensors(
    graph: ObservedBipartiteGraph,
    item_ids: torch.Tensor,
    user_ids: torch.Tensor,
    target_steps: torch.Tensor,
    target_clocks: torch.Tensor,
    k_max: int,
    k_min: int,
    beta_k: float,
    adaptive_k: bool,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, N = item_ids.shape
    users = torch.zeros((B, N, k_max), dtype=torch.long, device=device)
    lags = torch.zeros((B, N, k_max), dtype=torch.long, device=device)
    active_k = torch.full((B, N), k_max, dtype=torch.long, device=device)

    for b in range(B):
        c = int(target_clocks[b].item())
        target_event_step = int(target_steps[b].item())
        uid = int(user_ids[b].item())
        for n in range(N):
            iid = int(item_ids[b, n].item())
            degree = graph.get_item_degree(iid, c)
            if adaptive_k:
                k_act = min(k_max, max(k_min, int(math.ceil(beta_k * math.sqrt(degree + 1.0)))))
            else:
                k_act = k_max
            active_k[b, n] = k_act
            recs = graph.get_item_neighbor_users(
                iid,
                c=c,
                k_max=k_max,
                exclude_user=uid,
                exclude_event_step_ge=target_event_step,
            )
            for m, rec in enumerate(recs[:k_max]):
                users[b, n, m] = rec.user_id
                lags[b, n, m] = max(0, c - rec.event_clock)
    return users, lags, active_k


@torch.no_grad()
def evaluate(
    model: DGVDRec,
    loader: DataLoader,
    graph: ObservedBipartiteGraph,
    num_items: int,
    k_eval: int,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    recalls: List[float] = []
    ndcgs: List[float] = []
    for batch in loader:
        user_ids = batch["user_id"].to(device)
        seq_items = batch["seq_items"].to(device)
        seq_lags = batch["seq_lags"].to(device)
        seq_obs_age = batch["seq_obs_age"].to(device)
        observed_len = batch["observed_len"].to(device)
        missing_ratio = batch["missing_ratio"].to(device)
        delay_q = batch["delay_q"].to(device)
        target = batch["target"].to(device)
        target_step = batch["target_step"].to(device)
        target_clock = batch["target_clock"].to(device)
        eval_negs = batch["eval_negs"].to(device)
        if eval_negs.numel() == 0:
            raise ValueError("Evaluation dataset must be constructed with num_eval_neg > 0.")
        cands = torch.cat([target.unsqueeze(1), eval_negs], dim=1)

        h, lag_summary, _, _, _, _ = model.compute_user(
            user_ids, seq_items, seq_lags, seq_obs_age, observed_len, missing_ratio, delay_q, return_routing=True
        )
        neigh_u, neigh_lag, item_active_k = build_item_neighbor_tensors(
            graph, cands, user_ids, target_step, target_clock, model.k_max, model.k_min, model.beta_k, model.adaptive_k, device
        )
        scores = model.score_items(h, lag_summary, cands, neigh_u, neigh_lag, item_active_k, missing_ratio, delay_q)
        r, n = metrics_from_scores(scores, k=k_eval)
        recalls.append(r)
        ndcgs.append(n)
    return float(np.mean(recalls)), float(np.mean(ndcgs))


def train_model(
    data_root: str,
    dataset: str,
    user_col: str,
    item_col: str,
    time_col: str,
    max_len: int = 50,
    truncate_to: int = 200,
    batch_size: int = 256,
    n_epochs: int = 200,
    patience: int = 20,
    k_eval: int = 20,
    lr: float = 1e-3,
    lambda_l2: float = 1e-4,
    lambda_ent: float = 0.0,
    lambda_smooth: float = 0.0,
    d: int = 128,
    num_layers: int = 2,
    num_heads_var: int = 4,
    num_heads_time: int = 4,
    k_min: int = 5,
    k_max: int = 20,
    l_min: int = 5,
    lmax: int = 20,
    beta_k: float = 4.0,
    beta_l: float = 0.5,
    delay_quantile_p: float = 0.95,
    adaptive_k: bool = True,
    adaptive_l: bool = True,
    num_neg: int = 50,
    num_eval_neg: int = 100,
    num_pop_bins: int = 10,
    rank: int = 4,
    dropout: float = 0.2,
    delay_variant: str = "stationary",
    alpha: float = 1.0,
    delay_mu: float = 0.0,
    delay_sigma: float = 1.0,
    mu_alpha: float = 0.0,
    sigma_alpha: float = 0.35,
    time_block_size: int = 20,
    time_sigma: float = 0.35,
    seed: int = 42,
    clock_mode: str = "per_user",
    device: Optional[str] = None,
):
    set_seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    raw = load_raw_sequences(
        data_root=data_root,
        dataset=dataset,
        user_col=user_col,
        item_col=item_col,
        time_col=time_col,
        min_seq_len=3,
        min_cnt=5,
        truncate_to=truncate_to,
    )
    user2id, item2id = build_mappings(raw)
    delay_cfg = DelayConfig(
        alpha=alpha,
        mu=delay_mu,
        sigma=delay_sigma,
        lmax=lmax,
        variant=delay_variant,
        mu_alpha=mu_alpha,
        sigma_alpha=sigma_alpha,
        time_block_size=time_block_size,
        time_sigma=time_sigma,
        quantile_p=delay_quantile_p,
    )
    mapped = map_and_sample_delays(raw, user2id, item2id, delay_cfg, seed=seed, clock_mode=clock_mode)
    num_users = len(user2id)
    num_items = len(item2id)
    item_pop = build_item_pop(mapped, num_items)
    item2popbin = build_pop_bins(item_pop, num_bins=num_pop_bins, num_items=num_items)

    graph = ObservedBipartiteGraph.from_sequences(mapped, num_users, num_items, k_max=k_max)
    seen_by_user = build_seen_by_user(mapped)
    train_instances = build_instances(mapped, graph, split="train", max_len=max_len, delay_quantile_p=delay_quantile_p)
    val_instances = build_instances(mapped, graph, split="val", max_len=max_len, delay_quantile_p=delay_quantile_p)
    test_instances = build_instances(mapped, graph, split="test", max_len=max_len, delay_quantile_p=delay_quantile_p)

    def _avg_missing(instances: Sequence[CensoredInstance]) -> float:
        if not instances:
            return float("nan")
        return float(np.mean([inst.missing_ratio for inst in instances]))

    print(
        f"Loaded users={num_users}, items={num_items}, train={len(train_instances)}, "
        f"val={len(val_instances)}, test={len(test_instances)}, "
        f"delay_variant={delay_variant}, clock_mode={clock_mode}"
    )
    print(
        f"Avg missing ratio: train={_avg_missing(train_instances):.4f}, "
        f"val={_avg_missing(val_instances):.4f}, test={_avg_missing(test_instances):.4f}"
    )

    train_loader = DataLoader(CensoredSeqDataset(train_instances, max_len, num_items, seen_by_user), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        CensoredSeqDataset(val_instances, max_len, num_items, seen_by_user, num_eval_neg=num_eval_neg, seed=seed + 1000),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        CensoredSeqDataset(test_instances, max_len, num_items, seen_by_user, num_eval_neg=num_eval_neg, seed=seed + 2000),
        batch_size=batch_size,
        shuffle=False,
    )

    model = DGVDRec(
        num_users=num_users,
        num_items=num_items,
        max_len=max_len,
        d=d,
        num_pop_bins=num_pop_bins,
        item2popbin=item2popbin,
        rank=rank,
        num_heads_var=num_heads_var,
        num_heads_time=num_heads_time,
        num_layers=num_layers,
        k_min=k_min,
        k_max=k_max,
        l_min=l_min,
        lmax=lmax,
        beta_k=beta_k,
        beta_l=beta_l,
        adaptive_k=adaptive_k,
        adaptive_l=adaptive_l,
        dropout=dropout,
    ).to(dev)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lambda_l2)
    best_val = -1.0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    bad_epochs = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        losses: List[float] = []
        k_stats: List[float] = []
        l_stats: List[float] = []
        cand_stats: List[float] = []
        obs_pair_stats: List[float] = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            user_ids = batch["user_id"].to(dev)
            seq_items = batch["seq_items"].to(dev)
            seq_lags = batch["seq_lags"].to(dev)
            seq_obs_age = batch["seq_obs_age"].to(dev)
            observed_len = batch["observed_len"].to(dev)
            missing_ratio = batch["missing_ratio"].to(dev)
            delay_q = batch["delay_q"].to(dev)
            target = batch["target"].to(dev)
            target_step = batch["target_step"].to(dev)
            target_clock = batch["target_clock"].to(dev)

            neg = sample_negatives_train(user_ids, target, seen_by_user, num_items, num_neg, dev)
            cands = torch.cat([target.unsqueeze(1), neg], dim=1)

            h, lag_summary, _, ent, smooth, stats = model.compute_user(
                user_ids, seq_items, seq_lags, seq_obs_age, observed_len, missing_ratio, delay_q, return_routing=True
            )
            neigh_u, neigh_lag, item_active_k = build_item_neighbor_tensors(
                graph, cands, user_ids, target_step, target_clock, model.k_max, model.k_min, model.beta_k, model.adaptive_k, dev
            )
            scores = model.score_items(h, lag_summary, cands, neigh_u, neigh_lag, item_active_k, missing_ratio, delay_q)
            labels = torch.zeros(scores.size(0), dtype=torch.long, device=dev)
            loss = F.cross_entropy(scores, labels) + lambda_ent * ent + lambda_smooth * smooth

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            losses.append(float(loss.item()))
            k_stats.append(float(stats["active_k"].item()))
            l_stats.append(float(stats["active_l"].item()))
            cand_stats.append(float(stats["active_candidates"].item()))
            obs_pair_stats.append(float(stats["active_observed_pairs"].item()))

        val_hr, val_ndcg = evaluate(model, val_loader, graph, num_items, k_eval, dev)
        print(
            f"[Epoch {epoch:03d}] loss={np.mean(losses):.4f} val@{k_eval} "
            f"R={val_hr:.4f} NDCG={val_ndcg:.4f} "
            f"avgK={np.mean(k_stats):.2f} avgL={np.mean(l_stats):.2f} "
            f"avgBudget={np.mean(cand_stats):.2f} avgObservedPairs={np.mean(obs_pair_stats):.2f}"
        )

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
    test_hr, test_ndcg = evaluate(model, test_loader, graph, num_items, k_eval, dev)
    print(f"[Test@{k_eval}] R={test_hr:.4f} NDCG={test_ndcg:.4f}")
    return model, graph, user2id, item2id


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, choices=["toys", "beauty", "sports", "yelp"], default="beauty")
    p.add_argument("--data_root", type=str, default="./datasets/")
    p.add_argument("--max_len", type=int, default=50)
    p.add_argument("--truncate_to", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--n_epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--k_eval", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda_l2", type=float, default=1e-4)
    p.add_argument("--lambda_ent", type=float, default=0.0)
    p.add_argument("--lambda_smooth", type=float, default=0.0)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--num_heads_var", type=int, default=4)
    p.add_argument("--num_heads_time", type=int, default=4)
    p.add_argument("--k_min", type=int, default=5)
    p.add_argument("--k_max", type=int, default=20)
    p.add_argument("--l_min", type=int, default=5)
    p.add_argument("--lmax", type=int, default=20)
    p.add_argument("--beta_K", type=float, default=4.0)
    p.add_argument("--beta_L", type=float, default=0.5)
    p.add_argument("--delay_quantile_p", type=float, default=0.95)
    p.add_argument("--no_adaptive_K", action="store_true")
    p.add_argument("--no_adaptive_L", action="store_true")
    p.add_argument("--num_neg", type=int, default=50)
    p.add_argument("--num_eval_neg", type=int, default=100)
    p.add_argument("--num_pop_bins", type=int, default=10)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--delay_variant", type=str, choices=["stationary", "user", "time", "user_time"], default="stationary")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--delay_mu", type=float, default=0.0)
    p.add_argument("--delay_sigma", type=float, default=1.0)
    p.add_argument("--mu_alpha", type=float, default=0.0)
    p.add_argument("--sigma_alpha", type=float, default=0.35)
    p.add_argument("--time_block_size", type=int, default=20)
    p.add_argument("--time_sigma", type=float, default=0.35)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clock_mode", type=str, choices=["per_user", "global"], default="per_user")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def dataset_params(args: argparse.Namespace) -> Tuple[str, str, str, str, str]:
    if args.dataset == "toys":
        return args.data_root, "Toys.json", "reviewerID", "asin", "unixReviewTime"
    if args.dataset == "beauty":
        return args.data_root, "Beauty.json", "reviewerID", "asin", "unixReviewTime"
    if args.dataset == "sports":
        return args.data_root, "Sports.json", "reviewerID", "asin", "unixReviewTime"
    if args.dataset == "yelp":
        root = args.data_root if args.data_root != "./datasets/" else "./datasets/yelp"
        return root, "yelp_academic_dataset_review.json", "user_id", "business_id", "date"
    raise ValueError(f"Unknown dataset: {args.dataset}")


if __name__ == "__main__":
    args = parse_args()
    data_root, dataset_file, user_col, item_col, time_col = dataset_params(args)
    print(f"Dataset = {args.dataset} ({Path(data_root) / dataset_file})")
    print(
        f"DGVD aligned: d={args.d}, K=[{args.k_min},{args.k_max}], "
        f"L=[{args.l_min},{args.lmax}], beta_K={args.beta_K}, beta_L={args.beta_L}, "
        f"p={args.delay_quantile_p}, delay={args.delay_variant}, alpha={args.alpha}, "
        f"clock_mode={args.clock_mode}"
    )
    train_model(
        data_root=data_root,
        dataset=dataset_file,
        user_col=user_col,
        item_col=item_col,
        time_col=time_col,
        max_len=args.max_len,
        truncate_to=args.truncate_to,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        patience=args.patience,
        k_eval=args.k_eval,
        lr=args.lr,
        lambda_l2=args.lambda_l2,
        lambda_ent=args.lambda_ent,
        lambda_smooth=args.lambda_smooth,
        d=args.d,
        num_layers=args.num_layers,
        num_heads_var=args.num_heads_var,
        num_heads_time=args.num_heads_time,
        k_min=args.k_min,
        k_max=args.k_max,
        l_min=args.l_min,
        lmax=args.lmax,
        beta_k=args.beta_K,
        beta_l=args.beta_L,
        delay_quantile_p=args.delay_quantile_p,
        adaptive_k=not args.no_adaptive_K,
        adaptive_l=not args.no_adaptive_L,
        num_neg=args.num_neg,
        num_eval_neg=args.num_eval_neg,
        num_pop_bins=args.num_pop_bins,
        rank=args.rank,
        dropout=args.dropout,
        delay_variant=args.delay_variant,
        alpha=args.alpha,
        delay_mu=args.delay_mu,
        delay_sigma=args.delay_sigma,
        mu_alpha=args.mu_alpha,
        sigma_alpha=args.sigma_alpha,
        time_block_size=args.time_block_size,
        time_sigma=args.time_sigma,
        seed=args.seed,
        device=args.device,
    )
