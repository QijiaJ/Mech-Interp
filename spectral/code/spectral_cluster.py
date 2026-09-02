"""Exploratory event-level spectral clustering for L1H1 and L5H2.

This script implements the fixed coupling kernel in coupling_kernel.tex on the
already-frozen, source-disjoint natural corpus.  It does not fit reconstruction
operators and never accesses a locked or historical test split.

The four registered kernel variants isolate two concerns identified before the
run: attention-centred versus top-source-uniform-centred keys, and an additive
query block versus a fully conjunctive query-times-key-message kernel.  Labels
are used only after clustering for interpretation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import html
import importlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch


DTYPE = torch.float64
KERNEL_DTYPE = torch.float32
VARIANTS = (
    "add_attention_center",
    "add_uniform_top_center",
    "product_attention_center",
    "product_uniform_top_center",
)


@dataclass
class EventBank:
    target: str
    split: str
    q: torch.Tensor
    keys_attention_center: torch.Tensor
    keys_uniform_center: torch.Tensor
    messages: torch.Tensor
    weights: torch.Tensor
    source_ids: tuple[str, ...]
    query_positions: torch.Tensor
    semantic_labels: torch.Tensor
    subtype_labels: torch.Tensor
    families: tuple[str, ...]
    domains: tuple[str, ...]
    layout: torch.Tensor
    retained_attention: torch.Tensor
    retained_attention_squared: torch.Tensor

    def subset(self, indices: torch.Tensor) -> "EventBank":
        ids = indices.tolist()
        return EventBank(
            target=self.target,
            split=self.split,
            q=self.q.index_select(0, indices),
            keys_attention_center=self.keys_attention_center.index_select(0, indices),
            keys_uniform_center=self.keys_uniform_center.index_select(0, indices),
            messages=self.messages.index_select(0, indices),
            weights=self.weights.index_select(0, indices),
            source_ids=tuple(self.source_ids[i] for i in ids),
            query_positions=self.query_positions.index_select(0, indices),
            semantic_labels=self.semantic_labels.index_select(0, indices),
            subtype_labels=self.subtype_labels.index_select(0, indices),
            families=tuple(self.families[i] for i in ids),
            domains=tuple(self.domains[i] for i in ids),
            layout=self.layout.index_select(0, indices),
            retained_attention=self.retained_attention.index_select(0, indices),
            retained_attention_squared=self.retained_attention_squared.index_select(0, indices),
        )

    @property
    def count(self) -> int:
        return int(self.q.shape[0])


def _stable_key(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()


def _load_extraction_module(exploratory_root: Path, repo_root: Path) -> Any:
    for path in (
        exploratory_root / "code",
        repo_root,
        repo_root / "unifying_algorithm",
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    module = importlib.import_module("algorithm1")
    module._configure_imports(repo_root)
    return module


def _extract(
    args: argparse.Namespace,
) -> tuple[dict[str, list[Any]], Mapping[str, Any], Mapping[str, int], Mapping[str, Any]]:
    extraction = _load_extraction_module(args.exploratory_root, args.repo_root)
    from unifying_attention.experiments.unlearned_projector_real import (
        load_registered_model_and_tokenizer,
        resolve_registered_snapshot,
    )
    from unifying_attention.unlearned_projector_gate import require_valid_smoke_report

    require_valid_smoke_report(args.smoke_report)
    corpus = json.loads(args.corpus.read_text())
    artifacts = resolve_registered_snapshot(allow_download=False)
    if corpus["model"]["revision"] != artifacts.revision:
        raise RuntimeError("Frozen corpus and registered model revision differ.")
    model, tokenizer = load_registered_model_and_tokenizer(artifacts, allow_download=False)
    try:
        l1, audit1 = extraction._extract_head(
            model,
            tokenizer,
            corpus,
            target="L1H1",
            layer=1,
            head=1,
            chunk_size=args.chunk_size,
            batch_size=args.batch_size,
        )
        l5, audit5 = extraction._extract_head(
            model,
            tokenizer,
            corpus,
            target="L5H2",
            layer=5,
            head=2,
            chunk_size=args.chunk_size,
            batch_size=args.batch_size,
        )
    finally:
        del model
        del tokenizer
    c0 = {}
    for name, sources in (("L1H1", l1), ("L5H2", l5)):
        _v0, _complement, rank = extraction._fit_common_span(sources)
        c0[name] = rank
    return (
        {"L1H1": l1, "L5H2": l5},
        {"L1H1": audit1, "L5H2": audit5},
        c0,
        corpus,
    )


def _build_bank(
    sources: Sequence[Any],
    *,
    target: str,
    split: str,
    top_sources: int,
    max_events_per_source: int,
    event_rows: Mapping[tuple[str, int], Mapping[str, Any]],
) -> EventBank:
    q_rows: list[torch.Tensor] = []
    key_attention_rows: list[torch.Tensor] = []
    key_uniform_rows: list[torch.Tensor] = []
    message_rows: list[torch.Tensor] = []
    weight_rows: list[torch.Tensor] = []
    source_ids: list[str] = []
    query_positions: list[int] = []
    semantic_labels: list[int] = []
    subtype_labels: list[int] = []
    families: list[str] = []
    domains: list[str] = []
    layouts: list[torch.Tensor] = []
    retained_attention: list[float] = []
    retained_a2: list[float] = []

    for source in sources:
        if source.split != split:
            continue
        if source.complement_messages is None:
            raise RuntimeError("The common message span must be removed first.")
        candidates = []
        for event in range(source.event_count):
            mask = source.source_mask[event].clone()
            mask[0] = False
            positions = mask.nonzero(as_tuple=False).flatten()
            alpha = source.attention[event].index_select(0, positions).to(DTYPE)
            mu = source.complement_messages.index_select(0, positions).to(DTYPE)
            energy = float((alpha.square() * mu.square().sum(1)).sum())
            candidates.append((event, energy, int(source.query_positions[event])))
        candidates.sort(key=lambda row: (-row[1], row[2]))
        for event, _energy, query_position in candidates[:max_events_per_source]:
            mask = source.source_mask[event].clone()
            mask[0] = False
            valid = mask.nonzero(as_tuple=False).flatten()
            alpha_all = source.attention[event].index_select(0, valid).to(DTYPE)
            order = torch.argsort(alpha_all, descending=True, stable=True)
            chosen_local = order[:top_sources]
            chosen = valid.index_select(0, chosen_local)
            alpha = source.attention[event].index_select(0, chosen).to(DTYPE)
            raw_keys = source.keys.index_select(0, chosen).to(DTYPE)
            messages = source.complement_messages.index_select(0, chosen).to(DTYPE)

            all_keys = source.keys.index_select(0, valid).to(DTYPE)
            attention_center = (
                alpha_all[:, None] * all_keys
            ).sum(0) / alpha_all.sum().clamp_min(1e-30)
            uniform_center = raw_keys.mean(0)
            weights = alpha.square() / alpha_all.square().sum().clamp_min(1e-30)

            if len(chosen) < top_sources:
                pad = top_sources - len(chosen)
                raw_keys = torch.cat((raw_keys, torch.zeros(pad, raw_keys.shape[1], dtype=DTYPE)))
                messages = torch.cat((messages, torch.zeros(pad, messages.shape[1], dtype=DTYPE)))
                weights = torch.cat((weights, torch.zeros(pad, dtype=DTYPE)))

            q_rows.append(source.queries[event].to(DTYPE))
            key_attention_rows.append(raw_keys - attention_center)
            key_uniform_rows.append(raw_keys - uniform_center)
            message_rows.append(messages)
            weight_rows.append(weights)
            source_ids.append(str(source.source_id))
            query_positions.append(query_position)
            semantic_labels.append(int(source.semantic_labels[event]))
            subtype_labels.append(int(source.subtype_labels[event]))
            families.append(str(source.families[event]))
            domains.append(str(source.domain))
            metadata = event_rows[(str(source.source_id), query_position)]
            layouts.append(
                torch.tensor(
                    [
                        query_position / max(len(source.tokens) - 1, 1),
                        math.log(max(len(source.tokens), 2)),
                        int(metadata["minimum_target_distance"]) / max(len(source.tokens) - 1, 1),
                    ],
                    dtype=DTYPE,
                )
            )
            retained_attention.append(float(alpha.sum() / alpha_all.sum().clamp_min(1e-30)))
            retained_a2.append(float(alpha.square().sum() / alpha_all.square().sum().clamp_min(1e-30)))

    if not q_rows:
        raise RuntimeError(f"No {target} {split} events survived.")
    return EventBank(
        target=target,
        split=split,
        q=torch.stack(q_rows).to(KERNEL_DTYPE),
        keys_attention_center=torch.stack(key_attention_rows).to(KERNEL_DTYPE),
        keys_uniform_center=torch.stack(key_uniform_rows).to(KERNEL_DTYPE),
        messages=torch.stack(message_rows).to(KERNEL_DTYPE),
        weights=torch.stack(weight_rows).to(KERNEL_DTYPE),
        source_ids=tuple(source_ids),
        query_positions=torch.tensor(query_positions),
        semantic_labels=torch.tensor(semantic_labels),
        subtype_labels=torch.tensor(subtype_labels),
        families=tuple(families),
        domains=tuple(domains),
        layout=torch.stack(layouts),
        retained_attention=torch.tensor(retained_attention, dtype=DTYPE),
        retained_attention_squared=torch.tensor(retained_a2, dtype=DTYPE),
    )


def _keys(bank: EventBank, center: str) -> torch.Tensor:
    if center == "attention":
        return bank.keys_attention_center
    if center == "uniform_top":
        return bank.keys_uniform_center
    raise ValueError(center)


def _tensor_gram(
    left: EventBank,
    right: EventBank,
    *,
    center: str,
    block: int,
    symmetric: bool,
) -> torch.Tensor:
    lk = _keys(left, center)
    rk = _keys(right, center)
    lm = left.messages
    rm = right.messages
    lw = left.weights
    rw = right.weights
    result = torch.empty((left.count, right.count), dtype=KERNEL_DTYPE)
    for a0 in range(0, left.count, block):
        a1 = min(a0 + block, left.count)
        b_start = a0 if symmetric else 0
        for b0 in range(b_start, right.count, block):
            b1 = min(b0 + block, right.count)
            ka = lk[a0:a1].reshape(-1, lk.shape[-1])
            kb = rk[b0:b1].reshape(-1, rk.shape[-1])
            ma = lm[a0:a1].reshape(-1, lm.shape[-1])
            mb = rm[b0:b1].reshape(-1, rm.shape[-1])
            kd = (ka @ kb.T).reshape(a1 - a0, lk.shape[1], b1 - b0, rk.shape[1])
            md = (ma @ mb.T).reshape(a1 - a0, lm.shape[1], b1 - b0, rm.shape[1])
            values = (
                kd.square()
                * md.square()
                * lw[a0:a1, :, None, None]
                * rw[None, None, b0:b1, :]
            ).sum((1, 3))
            result[a0:a1, b0:b1] = values
            if symmetric and b0 != a0:
                result[b0:b1, a0:a1] = values.T
    return result.clamp_min_(0)


def _tensor_self(bank: EventBank, *, center: str) -> torch.Tensor:
    keys = _keys(bank, center)
    messages = bank.messages
    kd = torch.einsum("nmd,nkd->nmk", keys, keys)
    md = torch.einsum("nmd,nkd->nmk", messages, messages)
    return (
        kd.square()
        * md.square()
        * bank.weights[:, :, None]
        * bank.weights[:, None, :]
    ).sum((1, 2)).clamp_min(0)


def _normalised_kernel(
    q_cross: torch.Tensor,
    tensor_cross: torch.Tensor,
    left_q_self: torch.Tensor,
    right_q_self: torch.Tensor,
    left_tensor_self: torch.Tensor,
    right_tensor_self: torch.Tensor,
    *,
    mode: str,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mode == "add":
        raw = gamma * q_cross.square() + tensor_cross
        left_diag = gamma * left_q_self.square() + left_tensor_self
        right_diag = gamma * right_q_self.square() + right_tensor_self
    elif mode == "product":
        raw = q_cross.square() * tensor_cross
        left_diag = left_q_self.square() * left_tensor_self
        right_diag = right_q_self.square() * right_tensor_self
    else:
        raise ValueError(mode)
    floor = 1e-20
    denom = left_diag.clamp_min(floor).sqrt()[:, None] * right_diag.clamp_min(floor).sqrt()[None, :]
    kernel = (raw / denom).clamp(0, 1)
    return kernel.to(DTYPE), left_diag.to(DTYPE), right_diag.to(DTYPE)


def _adjusted_rand(labels_a: torch.Tensor, labels_b: torch.Tensor) -> float:
    a = labels_a.to(torch.long)
    b = labels_b.to(torch.long)
    ua, ia = torch.unique(a, sorted=True, return_inverse=True)
    ub, ib = torch.unique(b, sorted=True, return_inverse=True)
    table = torch.zeros((len(ua), len(ub)), dtype=DTYPE)
    table.index_put_((ia, ib), torch.ones(len(a), dtype=DTYPE), accumulate=True)
    comb = lambda x: (x * (x - 1) / 2).sum()
    index = comb(table)
    rows = comb(table.sum(1))
    cols = comb(table.sum(0))
    total = len(a) * (len(a) - 1) / 2
    if total <= 0:
        return 1.0
    expected = rows * cols / total
    maximum = 0.5 * (rows + cols)
    if abs(float(maximum - expected)) < 1e-30:
        return 1.0
    return float((index - expected) / (maximum - expected))


def _best_accuracy(truth: torch.Tensor, prediction: torch.Tensor) -> float:
    truth_values, t = torch.unique(truth, sorted=True, return_inverse=True)
    pred_values, p = torch.unique(prediction, sorted=True, return_inverse=True)
    size = max(len(truth_values), len(pred_values))
    table = torch.zeros((size, size), dtype=torch.long)
    table.index_put_((t, p), torch.ones(len(t), dtype=torch.long), accumulate=True)
    dp = {0: 0}
    for row in range(size):
        next_dp: dict[int, int] = {}
        for mask, value in dp.items():
            for column in range(size):
                if mask & (1 << column):
                    continue
                new_mask = mask | (1 << column)
                score = value + int(table[row, column])
                next_dp[new_mask] = max(next_dp.get(new_mask, -1), score)
        dp = next_dp
    return max(dp.values()) / max(len(truth), 1)


def _purity(truth: torch.Tensor, prediction: torch.Tensor) -> float:
    total = 0
    for label in torch.unique(prediction):
        selected = truth[prediction == label]
        total += int(torch.bincount(selected).max())
    return total / max(len(truth), 1)


def _kmeans(
    rows: torch.Tensor, components: int, *, restarts: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, float]:
    if components == 1:
        return torch.zeros(len(rows), dtype=torch.long), rows.mean(0, keepdim=True), 0.0
    best: tuple[torch.Tensor, torch.Tensor, float] | None = None
    for restart in range(restarts):
        generator = torch.Generator().manual_seed(seed + 1009 * restart)
        first = int(torch.randint(len(rows), (1,), generator=generator))
        center_indices = [first]
        centers = [rows[first]]
        distance = (rows - centers[0]).square().sum(1)
        for _ in range(1, components):
            if float(distance.sum()) <= 0:
                index = next(i for i in range(len(rows)) if i not in center_indices)
            else:
                index = int(torch.multinomial(distance / distance.sum(), 1, generator=generator))
            center_indices.append(index)
            centers.append(rows[index])
            distance = torch.minimum(distance, (rows - centers[-1]).square().sum(1))
        center = torch.stack(centers)
        labels = torch.full((len(rows),), -1, dtype=torch.long)
        for _ in range(100):
            proposed = torch.cdist(rows, center).square().argmin(1)
            if torch.equal(proposed, labels):
                break
            labels = proposed
            for unit in range(components):
                selected = labels == unit
                if not bool(selected.any()):
                    nearest = torch.cdist(rows, center).square().min(1).values
                    index = int(nearest.argmax())
                    labels[index] = unit
                    selected = labels == unit
                center[unit] = rows[selected].mean(0)
        inertia = float((rows - center.index_select(0, labels)).square().sum())
        if best is None or inertia < best[2]:
            best = labels.clone(), center.clone(), inertia
    assert best is not None
    return best


def _spectral(
    train_kernel: torch.Tensor,
    hold_kernel: torch.Tensor,
    *,
    max_a: int,
    restarts: int,
    seed: int,
    train_semantic: torch.Tensor,
    hold_semantic: torch.Tensor,
    train_subtype: torch.Tensor,
    hold_subtype: torch.Tensor,
) -> tuple[Mapping[str, Any], Mapping[int, tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    train = train_kernel.clone()
    train.fill_diagonal_(0)
    degree_train = train.sum(1).clamp_min(1e-20)
    affinity = train / degree_train.sqrt()[:, None] / degree_train.sqrt()[None, :]
    eigenvalues, eigenvectors = torch.linalg.eigh(affinity)
    order = torch.argsort(eigenvalues, descending=True)
    rho = eigenvalues.index_select(0, order)[: max_a + 1]
    vectors = eigenvectors.index_select(1, order)[:, :max_a]
    laplacian = 1 - rho
    gaps = {
        a: float(laplacian[a] - laplacian[a - 1])
        for a in range(1, max_a + 1)
    }
    selected_all = max(gaps, key=lambda a: (gaps[a], -a))
    selected_split = max(range(2, max_a + 1), key=lambda a: (gaps[a], -a))

    degree_hold = hold_kernel.sum(1).clamp_min(1e-20)
    cross_affinity = hold_kernel / degree_hold.sqrt()[:, None] / degree_train.sqrt()[None, :]
    results: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    rows = {}
    for components in range(1, max_a + 1):
        train_embedding = vectors[:, :components]
        train_embedding = train_embedding / train_embedding.norm(dim=1, keepdim=True).clamp_min(1e-30)
        labels, centers, inertia = _kmeans(
            train_embedding, components, restarts=restarts, seed=seed + components
        )
        hold_embedding = cross_affinity @ vectors[:, :components]
        hold_embedding = hold_embedding / rho[:components].clamp_min(1e-12)[None, :]
        hold_embedding = hold_embedding / hold_embedding.norm(dim=1, keepdim=True).clamp_min(1e-30)
        hold_distance = torch.cdist(hold_embedding, centers).square()
        hold_labels = hold_distance.argmin(1)
        if components > 1:
            sorted_distance = hold_distance.sort(1).values
            margin = sorted_distance[:, 1] - sorted_distance[:, 0]
            margin_record = {
                "median": float(margin.median()),
                "p10": float(torch.quantile(margin, 0.10)),
            }
        else:
            margin_record = {"median": 0.0, "p10": 0.0}
        rows[str(components)] = {
            "train_inertia": inertia,
            "train_occupancy": torch.bincount(labels, minlength=components).tolist(),
            "holdout_occupancy": torch.bincount(hold_labels, minlength=components).tolist(),
            "holdout_margin": margin_record,
            "train_semantic_ari": _adjusted_rand(train_semantic, labels),
            "holdout_semantic_ari": _adjusted_rand(hold_semantic, hold_labels),
            "holdout_semantic_accuracy": _best_accuracy(hold_semantic, hold_labels),
            "holdout_semantic_purity": _purity(hold_semantic, hold_labels),
            "train_subtype_ari": _adjusted_rand(train_subtype, labels),
            "holdout_subtype_ari": _adjusted_rand(hold_subtype, hold_labels),
            "holdout_subtype_accuracy": _best_accuracy(hold_subtype, hold_labels),
            "holdout_subtype_purity": _purity(hold_subtype, hold_labels),
        }
        results[components] = (labels, hold_labels)
    coverage = hold_kernel.max(1).values
    record = {
        "top_normalised_affinity_eigenvalues": rho.tolist(),
        "smallest_laplacian_eigenvalues": laplacian.tolist(),
        "eigengap_by_A": {str(key): value for key, value in gaps.items()},
        "selected_A_including_one": selected_all,
        "strongest_split_A": selected_split,
        "holdout_nearest_train_kernel": {
            "median": float(coverage.median()),
            "p10": float(torch.quantile(coverage, 0.10)),
        },
        "partitions": rows,
    }
    return record, results, vectors


def _spectral_gaps(kernel: torch.Tensor, *, max_a: int) -> Mapping[int, float]:
    """Return normalized-affinity eigengaps without fitting a partition."""
    affinity = kernel.clone()
    affinity.fill_diagonal_(0)
    degree = affinity.sum(1).clamp_min(1e-20)
    affinity = affinity / degree.sqrt()[:, None] / degree.sqrt()[None, :]
    eigenvalues = torch.linalg.eigvalsh(affinity).flip(0)[: max_a + 1]
    laplacian = 1 - eigenvalues
    return {
        a: float(laplacian[a] - laplacian[a - 1])
        for a in range(1, max_a + 1)
    }


def _event_alignment_null(
    bank: EventBank,
    *,
    max_a: int,
    events: int,
    repeats: int,
    seed: int,
    block: int,
) -> Mapping[str, Any]:
    """Permute complete key-message event blocks relative to the queries.

    The deterministic subset and permutations are label blind.  Unlike atom
    resampling, this preserves event centering, within-event source assembly,
    attention weights, and event norms.  It tests whether query/event alignment
    contributes more split structure than a chance alignment.
    """
    count = min(events, bank.count)
    order = sorted(
        range(bank.count),
        key=lambda i: _stable_key(bank.target, bank.source_ids[i], int(bank.query_positions[i])),
    )
    indices = torch.tensor(order[:count], dtype=torch.long)
    observed_bank = bank.subset(indices)
    q_cross = observed_bank.q @ observed_bank.q.T
    q_self = observed_bank.q.square().sum(1)

    tensor = _tensor_gram(
        observed_bank,
        observed_bank,
        center="attention",
        block=block,
        symmetric=True,
    )
    tensor_self = _tensor_self(observed_bank, center="attention")

    def kernel_for(candidate_tensor: torch.Tensor, candidate_self: torch.Tensor) -> torch.Tensor:
        kernel, _left, _right = _normalised_kernel(
            q_cross,
            candidate_tensor,
            q_self,
            q_self,
            candidate_self,
            candidate_self,
            mode="product",
            gamma=1.0,
        )
        kernel = 0.5 * (kernel + kernel.T)
        kernel.fill_diagonal_(1)
        return kernel

    observed = _spectral_gaps(kernel_for(tensor, tensor_self), max_a=max_a)
    null_rows: list[Mapping[int, float]] = []
    generator = torch.Generator().manual_seed(seed + 91373)
    for _ in range(repeats):
        permutation = torch.randperm(count, generator=generator)
        permuted_tensor = tensor.index_select(0, permutation).index_select(1, permutation)
        permuted_self = tensor_self.index_select(0, permutation)
        null_rows.append(
            _spectral_gaps(kernel_for(permuted_tensor, permuted_self), max_a=max_a)
        )

    by_a: dict[str, Any] = {}
    for a in range(1, max_a + 1):
        values = torch.tensor([row[a] for row in null_rows], dtype=DTYPE)
        obs = observed[a]
        by_a[str(a)] = {
            "observed_gap": obs,
            "null_median": float(values.median()),
            "null_q95": float(torch.quantile(values, 0.95)),
            "empirical_upper_p": (1 + int((values >= obs).sum())) / (repeats + 1),
        }
    observed_max_split = max(observed[a] for a in range(2, max_a + 1))
    null_max_split = torch.tensor(
        [max(row[a] for a in range(2, max_a + 1)) for row in null_rows],
        dtype=DTYPE,
    )
    return {
        "purpose": "label-blind exploratory calibration; not a formal model-selection test",
        "events": count,
        "repeats": repeats,
        "null": "retain q; permute complete centered-key/message event blocks, including their internal weights, relative to q",
        "eigengap_by_A": by_a,
        "max_split_gap_A2_to_Amax": {
            "observed": observed_max_split,
            "null_median": float(null_max_split.median()),
            "null_q95": float(torch.quantile(null_max_split, 0.95)),
            "empirical_upper_p": (
                1 + int((null_max_split >= observed_max_split).sum())
            ) / (repeats + 1),
        },
    }


def _kernel_pca_coverage(train_kernel: torch.Tensor, hold_kernel: torch.Tensor) -> Mapping[str, Any]:
    values, vectors = torch.linalg.eigh(train_kernel)
    order = torch.argsort(values, descending=True)
    values = values.index_select(0, order).clamp_min(0)
    vectors = vectors.index_select(1, order)
    total = values.sum().clamp_min(1e-30)
    rank = int((values.cumsum(0) < 0.90 * total).sum()) + 1
    positive = values[:rank] > 1e-12 * values[0].clamp_min(1e-30)
    rank = int(positive.sum())
    coordinates = hold_kernel @ vectors[:, :rank]
    capture = (coordinates.square() / values[:rank].clamp_min(1e-30)[None, :]).sum(1)
    capture = capture.clamp(0, 1)
    return {
        "train_rank_90": rank,
        "holdout_capture_median": float(capture.median()),
        "holdout_capture_p10": float(torch.quantile(capture, 0.10)),
        "holdout_capture_mean": float(capture.mean()),
    }


def _matched_layout_indices(
    bank: EventBank,
    *,
    mean: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, Mapping[str, Any]]:
    labels = bank.semantic_labels
    counts = torch.bincount(labels)
    minority = int(counts.argmin())
    majority = int(counts.argmax())
    minor = (labels == minority).nonzero(as_tuple=False).flatten().tolist()
    available = set((labels == majority).nonzero(as_tuple=False).flatten().tolist())
    z = (bank.layout - mean) / scale
    selected: list[int] = []
    distances: list[float] = []
    for index in minor:
        same_domain = [j for j in available if bank.domains[j] == bank.domains[index]]
        candidates = same_domain or sorted(available)
        if not candidates:
            break
        chosen = min(
            candidates,
            key=lambda j: (float((z[index] - z[j]).square().sum()), j),
        )
        available.remove(chosen)
        selected.extend((index, chosen))
        distances.append(float((z[index] - z[chosen]).norm()))
    indices = torch.tensor(sorted(selected), dtype=torch.long)
    return indices, {
        "minority_label": minority,
        "pairs": len(distances),
        "same_domain_pairs": sum(
            bank.domains[selected[2 * i]] == bank.domains[selected[2 * i + 1]]
            for i in range(len(distances))
        ),
        "standardised_layout_distance_median": (
            float(torch.tensor(distances).median()) if distances else None
        ),
    }


def _matched_layout_audit(
    train: EventBank,
    hold: EventBank,
    train_kernel: torch.Tensor,
    hold_kernel: torch.Tensor,
    *,
    restarts: int,
    seed: int,
) -> Mapping[str, Any]:
    mean = train.layout.mean(0)
    scale = train.layout.std(0).clamp_min(1e-8)
    train_indices, train_match = _matched_layout_indices(train, mean=mean, scale=scale)
    hold_indices, hold_match = _matched_layout_indices(hold, mean=mean, scale=scale)
    local_train = train_kernel.index_select(0, train_indices).index_select(1, train_indices)
    local_hold = hold_kernel.index_select(0, hold_indices).index_select(1, train_indices)
    spectral, _partitions, _vectors = _spectral(
        local_train,
        local_hold,
        max_a=2,
        restarts=restarts,
        seed=seed + 70001,
        train_semantic=train.semantic_labels.index_select(0, train_indices),
        hold_semantic=hold.semantic_labels.index_select(0, hold_indices),
        train_subtype=train.subtype_labels.index_select(0, train_indices),
        hold_subtype=hold.subtype_labels.index_select(0, hold_indices),
    )
    return {
        "purpose": "post-fit diagnostic only; biological labels form matched subsets but never fit the full partition",
        "layout_coordinates": [
            "normalised_query_position",
            "log_sequence_length",
            "normalised_minimum_target_distance",
        ],
        "train_matching": train_match,
        "holdout_matching": hold_match,
        "spectral": spectral,
    }


def _svg_scatter(
    path: Path,
    coordinates: torch.Tensor,
    discovered: torch.Tensor,
    biological: torch.Tensor,
    *,
    title: str,
) -> None:
    width, height = 920, 430
    panels = ((discovered, "discovered A=2"), (biological, "post-hoc biological label"))
    colors = ("#2d6cdf", "#e4572e", "#2ca02c", "#9467bd", "#8c564b")
    xy = coordinates[:, :2].to(DTYPE)
    low = xy.quantile(0.01, dim=0)
    high = xy.quantile(0.99, dim=0)
    span = (high - low).clamp_min(1e-12)
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="24" y="28" font-family="sans-serif" font-size="17" font-weight="600">{html.escape(title)}</text>',
    ]
    for panel, (labels, label) in enumerate(panels):
        x0 = 35 + panel * 455
        y0 = 55
        pw, ph = 410, 340
        chunks.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="#fafafa" stroke="#bbb"/>')
        chunks.append(f'<text x="{x0}" y="{height-12}" font-family="sans-serif" font-size="13">{html.escape(label)}</text>')
        for index in range(len(xy)):
            px = x0 + 8 + float((xy[index, 0] - low[0]) / span[0]) * (pw - 16)
            py = y0 + ph - 8 - float((xy[index, 1] - low[1]) / span[1]) * (ph - 16)
            color = colors[int(labels[index]) % len(colors)]
            chunks.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="1.6" fill="{color}" fill-opacity="0.55"/>')
    chunks.append('</svg>')
    path.write_text("\n".join(chunks) + "\n")


def _analyse_head(
    sources: Sequence[Any],
    *,
    target: str,
    args: argparse.Namespace,
    plot_dir: Path,
    corpus: Mapping[str, Any],
) -> Mapping[str, Any]:
    event_rows = {
        (str(row["source_id"]), int(row["query_position"])): row
        for row in corpus[target]["events"]
    }
    train = _build_bank(
        sources,
        target=target,
        split="train",
        top_sources=args.top_sources,
        max_events_per_source=args.max_events_per_source,
        event_rows=event_rows,
    )
    hold = _build_bank(
        sources,
        target=target,
        split="holdout",
        top_sources=args.top_sources,
        max_events_per_source=args.max_events_per_source,
        event_rows=event_rows,
    )
    q_train_cross = train.q @ train.q.T
    q_hold_cross = hold.q @ train.q.T
    q_train_self = train.q.square().sum(1)
    q_hold_self = hold.q.square().sum(1)
    variants: dict[str, Any] = {}
    retained: dict[str, tuple[torch.Tensor, torch.Tensor, Mapping[int, tuple[torch.Tensor, torch.Tensor]], torch.Tensor]] = {}
    for center in ("attention", "uniform_top"):
        tensor_train = _tensor_gram(
            train, train, center=center, block=args.kernel_block, symmetric=True
        )
        tensor_hold = _tensor_gram(
            hold, train, center=center, block=args.kernel_block, symmetric=False
        )
        tensor_train_self = _tensor_self(train, center=center)
        tensor_hold_self = _tensor_self(hold, center=center)
        gamma = float(
            tensor_train_self.to(DTYPE).median()
            / q_train_self.to(DTYPE).square().median().clamp_min(1e-30)
        )
        for mode in ("add", "product"):
            name = f"{mode}_{'attention_center' if center == 'attention' else 'uniform_top_center'}"
            train_kernel, _left_diag, _right_diag = _normalised_kernel(
                q_train_cross,
                tensor_train,
                q_train_self,
                q_train_self,
                tensor_train_self,
                tensor_train_self,
                mode=mode,
                gamma=gamma,
            )
            train_kernel = 0.5 * (train_kernel + train_kernel.T)
            train_kernel.fill_diagonal_(1)
            hold_kernel, _hold_diag, _train_diag = _normalised_kernel(
                q_hold_cross,
                tensor_hold,
                q_hold_self,
                q_train_self,
                tensor_hold_self,
                tensor_train_self,
                mode=mode,
                gamma=gamma,
            )
            spectral, partitions, vectors = _spectral(
                train_kernel,
                hold_kernel,
                max_a=args.max_a,
                restarts=args.kmeans_restarts,
                seed=args.seed,
                train_semantic=train.semantic_labels,
                hold_semantic=hold.semantic_labels,
                train_subtype=train.subtype_labels,
                hold_subtype=hold.subtype_labels,
            )
            spectral["gamma"] = gamma
            variants[name] = spectral
            retained[name] = (train_kernel, hold_kernel, partitions, vectors)

    largest_split_gap = max(
        VARIANTS,
        key=lambda name: (
            variants[name]["eigengap_by_A"][str(variants[name]["strongest_split_A"])],
            variants[name]["holdout_nearest_train_kernel"]["median"],
        ),
    )
    primary = "product_attention_center"
    train_kernel, hold_kernel, partitions, vectors = retained[primary]
    variants[primary]["one_subspace_coverage"] = _kernel_pca_coverage(train_kernel, hold_kernel)
    variants[primary]["event_alignment_null"] = _event_alignment_null(
        train,
        max_a=args.max_a,
        events=args.null_events,
        repeats=args.null_repeats,
        seed=args.seed,
        block=args.kernel_block,
    )
    if target == "L1H1":
        variants[primary]["matched_layout_audit"] = _matched_layout_audit(
            train,
            hold,
            train_kernel,
            hold_kernel,
            restarts=args.kmeans_restarts,
            seed=args.seed,
        )
    plot_labels = partitions[2][0]
    biological = train.semantic_labels if target == "L1H1" else train.subtype_labels
    plot_path = plot_dir / f"{target}_spectral_embedding.svg"
    _svg_scatter(
        plot_path,
        vectors[:, 1:3] if vectors.shape[1] >= 3 else vectors[:, :2],
        plot_labels,
        biological,
        title=f"{target}: {primary}",
    )
    return {
        "target": target,
        "train_events": train.count,
        "holdout_events": hold.count,
        "train_sources": len(set(train.source_ids)),
        "holdout_sources": len(set(hold.source_ids)),
        "train_family_counts": dict(sorted(Counter(train.families).items())),
        "holdout_family_counts": dict(sorted(Counter(hold.families).items())),
        "top_source_retention": {
            "train_attention_median": float(train.retained_attention.median()),
            "train_attention_squared_median": float(train.retained_attention_squared.median()),
            "holdout_attention_median": float(hold.retained_attention.median()),
            "holdout_attention_squared_median": float(hold.retained_attention_squared.median()),
        },
        "primary_variant": primary,
        "primary_variant_reason": "same fully conjunctive Q-times-key-message kernel for both heads",
        "largest_split_gap_variant": largest_split_gap,
        "plot": str(plot_path),
        "variants": variants,
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    sources, extraction_audit, common_rank, corpus = _extract(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "analysis": "event_level_coupling_kernel_spectral_clustering",
        "development_only": True,
        "reconstruction_run": False,
        "historical_test_accessed": False,
        "locked_test_accessed": False,
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "smoke_report": str(args.smoke_report),
        "configuration": {
            "top_sources": args.top_sources,
            "max_events_per_source": args.max_events_per_source,
            "event_selection": "highest exact non-BOS pair-write energy within each source; labels hidden",
            "source_weights": "alpha^2 divided by complete non-BOS row alpha^2 sum",
            "kernel_variants": list(VARIANTS),
            "max_A": args.max_a,
            "kmeans_restarts": args.kmeans_restarts,
            "kernel_block": args.kernel_block,
            "null_events": args.null_events,
            "null_repeats": args.null_repeats,
            "seed": args.seed,
            "common_message_rank_90": common_rank,
        },
        "extraction_audit": extraction_audit,
        "L1H1": _analyse_head(
            sources["L1H1"],
            target="L1H1",
            args=args,
            plot_dir=args.output_dir,
            corpus=corpus,
        ),
        "L5H2": _analyse_head(
            sources["L5H2"],
            target="L5H2",
            args=args,
            plot_dir=args.output_dir,
            corpus=corpus,
        ),
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve()
    exploratory = here.parents[2] if here.parent.name == "code" else here.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--exploratory-root", type=Path, default=exploratory)
    parser.add_argument("--smoke-report", type=Path, required=True)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=exploratory / "data" / "natural_behavior_corpus.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--top-sources", type=int, default=8)
    parser.add_argument("--max-events-per-source", type=int, default=6)
    parser.add_argument("--kernel-block", type=int, default=48)
    parser.add_argument("--max-a", type=int, default=4)
    parser.add_argument("--kmeans-restarts", type=int, default=20)
    parser.add_argument("--null-events", type=int, default=400)
    parser.add_argument("--null-repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1729)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    numeric = (
        args.chunk_size,
        args.batch_size,
        args.threads,
        args.top_sources,
        args.max_events_per_source,
        args.kernel_block,
        args.max_a,
        args.kmeans_restarts,
        args.null_events,
        args.null_repeats,
    )
    if min(numeric) <= 0:
        raise SystemExit("All numerical controls must be positive.")
    if args.max_a < 2:
        raise SystemExit("max-a must include at least the A=2 diagnostic.")
    torch.set_num_threads(args.threads)
    result = run(args)
    output = args.output_dir / "spectral_results.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
