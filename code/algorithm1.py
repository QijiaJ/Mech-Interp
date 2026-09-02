"""Minimal learned rank-one pair co-clustering experiment.

This development script fits one positive Q/K map per target head on the
annotated TRAIN attention rows, freezes it, and factorizes every non-BOS
query--source pair X_gj = p_gj m_gj^T with a hard joint (U_a, V_a) label.
The fitted objective is pair reconstruction only.  The principal output metric
is the residual-stream readout of the reconstructed rank-one matrices.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import torch


DTYPE = torch.float64
FIT_DTYPE = torch.float32
EPSILON = 1e-12
WIDTH = 512
TOP_K = 64


def _configure_imports(repo_root: Path) -> None:
    for path in (repo_root, repo_root / "unifying_algorithm"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _stable_key(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()


@dataclass
class SourceRows:
    source_id: str
    split: str
    domain: str
    tokens: tuple[str, ...]
    queries: torch.Tensor                 # [E,d]
    keys: torch.Tensor                    # [T,d]
    messages: torch.Tensor                # [T,dm]
    attention: torch.Tensor               # [E,T]
    source_mask: torch.Tensor             # [E,T]
    query_positions: torch.Tensor         # [E]
    semantic_labels: torch.Tensor         # [E]
    subtype_labels: torch.Tensor          # [E]
    families: tuple[str, ...]
    background_writes: torch.Tensor       # [B,dm], TRAIN only
    query_codes: torch.Tensor | None = None
    key_codes: torch.Tensor | None = None
    complement_messages: torch.Tensor | None = None

    @property
    def event_count(self) -> int:
        return int(self.queries.shape[0])


@dataclass
class FitState:
    relation: tuple[torch.Tensor, ...]
    message: tuple[torch.Tensor, ...]
    assignments: tuple[torch.Tensor, ...]
    eta_x: float
    history: tuple[Mapping[str, Any], ...]
    converged: bool


def _pad_sources(
    sources: Sequence[tuple[str, tuple[int, ...]]], pad_token_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(ids) for _source_id, ids in sources], dtype=torch.long)
    width = int(lengths.max())
    result = torch.full((len(sources), width), int(pad_token_id), dtype=torch.long)
    for row, (_source_id, ids) in enumerate(sources):
        result[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return result, lengths


def _tokenize_source(tokenizer: Any, row: Mapping[str, Any]) -> tuple[tuple[int, ...], tuple[str, ...]]:
    encoded = tokenizer(row["text"], add_special_tokens=True)
    ids = tuple(int(value) for value in encoded["input_ids"])
    tokens = tuple(tokenizer.decode([value], skip_special_tokens=False) for value in ids)
    if "".join(tokens) != "<bos>" + row["text"]:
        raise RuntimeError(f"Tokenizer round trip failed for source {row['source_id']}.")
    return ids, tokens


def _extract_head(
    model: Any,
    tokenizer: Any,
    corpus: Mapping[str, Any],
    *,
    target: str,
    layer: int,
    head: int,
    chunk_size: int,
    batch_size: int,
) -> tuple[list[SourceRows], Mapping[str, float]]:
    from unifying_attention.gemma import extract_layer_qk
    from unifying_attention.unlearned_projector_data import compute_message_basis, head_output_block

    if target == "L1H1":
        semantic = {
            "alphabetic_word": 0,
            "four_digit_year": 0,
            "identifier": 0,
            "other_number": 0,
            "newline_boundary": 1,
        }
        subtype_names = tuple(sorted(semantic))
        subtype = {name: index for index, name in enumerate(subtype_names)}
    else:
        semantic = {"strict_induction": 0, "tokenization_tolerant_induction": 0}
        subtype = {"strict_induction": 0, "tokenization_tolerant_induction": 1}

    source_rows = {str(row["source_id"]): row for row in corpus["sources"]}
    tokenized = {
        source_id: _tokenize_source(tokenizer, row)
        for source_id, row in source_rows.items()
    }
    events_by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for event in corpus[target]["events"]:
        events_by_source[str(event["source_id"])].append(event)

    basis = compute_message_basis(
        head_output_block(model, layer_idx=layer, head=head), expected_rank=256
    ).basis
    extracted: list[SourceRows] = []
    audit_max: dict[str, float] = defaultdict(float)
    for split in ("train", "holdout"):
        selected = sorted(
            (
                (source_id, tokenized[source_id][0])
                for source_id, row in source_rows.items()
                if row["split"] == split
            ),
            key=lambda item: int(item[0]),
        )
        for start in range(0, len(selected), chunk_size):
            chunk = selected[start : start + chunk_size]
            ids, lengths = _pad_sources(chunk, tokenizer.pad_token_id)
            data, audit = extract_layer_qk(
                model,
                input_ids=ids,
                prompt_lengths=lengths,
                layer_idx=layer,
                batch_size=batch_size,
                recompute_ov_float64=True,
            )
            for name, value in audit.items():
                if isinstance(value, (int, float)):
                    audit_max[name] = max(audit_max[name], abs(float(value)))
            for batch_index, (source_id, source_ids) in enumerate(chunk):
                length = len(source_ids)
                rows = sorted(
                    events_by_source[source_id], key=lambda row: int(row["query_position"])
                )
                if not rows:
                    continue
                queries = []
                attentions = []
                masks = []
                query_positions = []
                semantic_labels = []
                subtype_labels = []
                families = []
                for row in rows:
                    if row["split"] != split:
                        raise RuntimeError("Event/source split mismatch.")
                    query = int(row["query_position"])
                    family = str(row["family"])
                    queries.append(data.queries[batch_index, head, query].to(FIT_DTYPE).cpu())
                    attentions.append(data.attention[batch_index, head, query, :length].to(FIT_DTYPE).cpu())
                    masks.append(data.source_mask[batch_index, query, :length].cpu())
                    query_positions.append(query)
                    semantic_labels.append(semantic[family])
                    subtype_labels.append(subtype[family])
                    families.append(family)

                keys = data.keys[batch_index, head, :length].to(FIT_DTYPE).cpu()
                messages = (
                    data.messages[batch_index, head, :length].to(DTYPE) @ basis
                ).to(FIT_DTYPE).cpu()
                annotated = set(query_positions)
                candidates = [
                    query
                    for query in range(1, length - 1)
                    if query not in annotated and bool(data.valid_queries[batch_index, query])
                ]
                candidates.sort(key=lambda query: _stable_key(target, source_id, query))
                background = []
                if split == "train":
                    for query in candidates[:8]:
                        alpha = data.attention[batch_index, head, query, 1:length].to(DTYPE)
                        background.append((alpha[:, None] * messages[1:].to(DTYPE)).sum(0))
                extracted.append(
                    SourceRows(
                        source_id=source_id,
                        split=split,
                        domain=str(source_rows[source_id]["domain"]),
                        tokens=tokenized[source_id][1],
                        queries=torch.stack(queries),
                        keys=keys,
                        messages=messages,
                        attention=torch.stack(attentions),
                        source_mask=torch.stack(masks),
                        query_positions=torch.tensor(query_positions, dtype=torch.long),
                        semantic_labels=torch.tensor(semantic_labels, dtype=torch.long),
                        subtype_labels=torch.tensor(subtype_labels, dtype=torch.long),
                        families=tuple(families),
                        background_writes=(
                            torch.stack(background).to(FIT_DTYPE)
                            if background
                            else torch.empty((0, 256), dtype=FIT_DTYPE)
                        ),
                    )
                )
            del data
    return extracted, dict(audit_max)


def _positive_map(input_dim: int, seed: int) -> Any:
    from unifying_attention.qk import LayerGlobalPositiveQK

    torch.manual_seed(seed)
    return LayerGlobalPositiveQK(
        input_dim=input_dim, width=WIDTH, top_k=TOP_K, epsilon=EPSILON
    )


def _source_attention_kl(model: Any, source: SourceRows) -> torch.Tensor:
    q = model.encode_queries(source.queries)
    k = model.encode_keys(source.keys)
    kernel = q @ k.T + EPSILON
    kernel = kernel * source.source_mask.to(kernel.dtype)
    predicted = kernel / kernel.sum(1, keepdim=True).clamp_min(EPSILON)
    exact = source.attention
    return (
        exact
        * (exact.clamp_min(EPSILON).log() - predicted.clamp_min(EPSILON).log())
    ).sum(1)


@torch.no_grad()
def _mean_attention_kl(model: Any, sources: Sequence[SourceRows]) -> float:
    total = 0.0
    count = 0
    for source in sources:
        rows = _source_attention_kl(model, source)
        total += float(rows.sum())
        count += len(rows)
    return total / max(count, 1)


def _fit_feature_map(
    sources: Sequence[SourceRows], *, seed: int, epochs: int, learning_rate: float
) -> tuple[Any, Mapping[str, Any]]:
    train = [source for source in sources if source.split == "train"]
    holdout = [source for source in sources if source.split == "holdout"]
    model = _positive_map(train[0].queries.shape[1], seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed + 81121)
    initial = _mean_attention_kl(model, train)
    best = initial
    best_epoch = 0
    best_state = deepcopy(model.state_dict())
    model.train()
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(train), generator=generator).tolist()
        for index in order:
            loss = _source_attention_kl(model, train[index]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        value = _mean_attention_kl(model, train)
        if value < best:
            best = value
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
        model.train()
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(json.dumps({"stage": "feature-map", "epoch": epoch, "train_kl": value}), flush=True)
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        for source in sources:
            source.query_codes = model.encode_queries(source.queries).detach().cpu()
            source.key_codes = model.encode_keys(source.keys).detach().cpu()
    return model, {
        "seed": seed,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "initial_train_attention_kl": initial,
        "best_epoch": best_epoch,
        "train_attention_kl": _mean_attention_kl(model, train),
        "holdout_attention_kl": _mean_attention_kl(model, holdout),
    }


def _fit_common_span(sources: Sequence[SourceRows], energy: float = 0.90) -> tuple[torch.Tensor, torch.Tensor, int]:
    rows = torch.cat(
        [source.background_writes for source in sources if source.split == "train"], dim=0
    ).to(DTYPE)
    _u, singular, vh = torch.linalg.svd(rows, full_matrices=False)
    spectrum = singular.square()
    rank = int((spectrum.cumsum(0) < energy * spectrum.sum()).sum()) + 1
    v0 = vh[:rank].T.contiguous()
    identity = torch.eye(rows.shape[1], dtype=DTYPE)
    complete = torch.linalg.qr(torch.cat((v0, identity), dim=1), mode="complete").Q
    complement = complete[:, rank:]
    for source in sources:
        source.complement_messages = (source.messages.to(DTYPE) @ complement).to(FIT_DTYPE)
    return v0, complement, rank


def _pair_block(source: SourceRows) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if source.query_codes is None or source.key_codes is None or source.complement_messages is None:
        raise RuntimeError("Source has not been prepared.")
    kernel = source.query_codes @ source.key_codes.T + EPSILON
    kernel = kernel * source.source_mask.to(kernel.dtype)
    denominator = kernel.sum(1).clamp_min(EPSILON)
    product = source.query_codes[:, None, :] * source.key_codes[None, :, :]
    addresses = product / denominator[:, None, None]
    nonbos = source.source_mask.clone()
    nonbos[:, 0] = False
    event, position = nonbos.nonzero(as_tuple=True)
    return (
        addresses[event, position].to(DTYPE),
        source.complement_messages.index_select(0, position).to(DTYPE),
        source.attention[event, position].to(DTYPE),
        event,
        position,
    )


def _event_pair_energies(sources: Sequence[SourceRows]) -> tuple[list[torch.Tensor], float, int]:
    energies = []
    all_event = []
    event_count = 0
    for source in sources:
        p, m, alpha, event, _position = _pair_block(source)
        pair = alpha.square() * p.square().sum(1) * m.square().sum(1)
        local = torch.zeros(source.event_count, dtype=DTYPE).index_add_(0, event, pair)
        energies.append(local)
        all_event.append(local)
        event_count += source.event_count
    concatenated = torch.cat(all_event)
    positive = concatenated[concatenated > 0]
    eta = 1e-8 * float(positive.median()) if len(positive) else 1e-8
    return energies, eta, event_count


def _weights_for_source(
    source: SourceRows, event_energy: torch.Tensor, eta: float, event_count: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    p, m, alpha, event, position = _pair_block(source)
    weight = alpha.square() / (
        float(event_count) * (event_energy + eta).index_select(0, event)
    )
    return p, m, alpha, event, position, weight


def _weighted_stats(sources: Sequence[SourceRows]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_sum = torch.zeros(256, dtype=DTYPE)
    k_sum = torch.zeros(256, dtype=DTYPE)
    q_outer = torch.zeros(256, 256, dtype=DTYPE)
    k_outer = torch.zeros(256, 256, dtype=DTYPE)
    count = 0
    for source in sources:
        mask = source.source_mask.clone()
        mask[:, 0] = False
        event, position = mask.nonzero(as_tuple=True)
        q = source.queries.index_select(0, event).to(DTYPE)
        k = source.keys.index_select(0, position).to(DTYPE)
        q_sum += q.sum(0)
        k_sum += k.sum(0)
        q_outer += q.T @ q
        k_outer += k.T @ k
        count += len(q)
    q_mean = q_sum / count
    k_mean = k_sum / count
    q_cov = q_outer / count - q_mean[:, None] * q_mean[None, :]
    k_cov = k_outer / count - k_mean[:, None] * k_mean[None, :]
    q_basis = torch.linalg.eigh(q_cov).eigenvectors[:, -3:]
    k_basis = torch.linalg.eigh(k_cov).eigenvectors[:, -3:]
    return q_mean, k_mean, q_basis, k_basis


def _kmeans(coordinates: torch.Tensor, components: int, seed: int) -> torch.Tensor:
    if components == 1:
        return torch.zeros(len(coordinates), dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    centers = [coordinates[int(torch.randint(len(coordinates), (1,), generator=generator))]]
    distance = (coordinates - centers[0]).square().sum(1)
    for _ in range(1, components):
        probabilities = distance / distance.sum().clamp_min(1e-30)
        index = int(torch.multinomial(probabilities, 1, generator=generator))
        centers.append(coordinates[index])
        distance = torch.minimum(distance, (coordinates - centers[-1]).square().sum(1))
    centers_t = torch.stack(centers)
    labels = torch.zeros(len(coordinates), dtype=torch.long)
    for _ in range(20):
        proposed = torch.cdist(coordinates, centers_t).square().argmin(1)
        if torch.equal(proposed, labels):
            break
        labels = proposed
        for unit in range(components):
            selected = labels == unit
            if not bool(selected.any()):
                farthest = int(torch.cdist(coordinates, centers_t).square().min(1).values.argmax())
                labels[farthest] = unit
                selected = labels == unit
            centers_t[unit] = coordinates[selected].mean(0)
    return labels


def _initial_assignments(
    sources: Sequence[SourceRows], components: int, seed: int
) -> tuple[torch.Tensor, ...]:
    q_mean, k_mean, q_basis, k_basis = _weighted_stats(sources)
    blocks = []
    lengths = []
    for source in sources:
        mask = source.source_mask.clone()
        mask[:, 0] = False
        event, position = mask.nonzero(as_tuple=True)
        q = (source.queries.index_select(0, event).to(DTYPE) - q_mean) @ q_basis
        k = (source.keys.index_select(0, position).to(DTYPE) - k_mean) @ k_basis
        block = torch.cat((q, k), dim=1)
        blocks.append(block)
        lengths.append(len(block))
    coordinates = torch.cat(blocks)
    rms = coordinates.square().mean(0).sqrt().clamp_min(1e-12)
    labels = _kmeans((coordinates / rms).to(FIT_DTYPE), components, seed)
    return tuple(labels.split(lengths))


def _operator(
    sources: Sequence[SourceRows],
    assignments: Sequence[torch.Tensor],
    event_energies: Sequence[torch.Tensor],
    eta: float,
    event_count: int,
    unit: int,
    side: str,
    opposite: torch.Tensor | None,
) -> tuple[int, Any]:
    dimension = WIDTH if side == "relation" else int(sources[0].complement_messages.shape[1])

    def apply(vectors: torch.Tensor) -> torch.Tensor:
        output = torch.zeros(dimension, vectors.shape[1], dtype=DTYPE)
        for source, labels, energy in zip(sources, assignments, event_energies):
            p, m, _alpha, _event, _position, weight = _weights_for_source(
                source, energy, eta, event_count
            )
            selected = labels == unit
            if not bool(selected.any()):
                continue
            p = p[selected]
            m = m[selected]
            weight = weight[selected]
            if side == "relation":
                coefficient = weight * (
                    m.square().sum(1)
                    if opposite is None
                    else (m @ opposite).square().sum(1)
                )
                output += p.T @ (coefficient[:, None] * (p @ vectors))
            else:
                coefficient = weight * (
                    p.square().sum(1)
                    if opposite is None
                    else (p @ opposite).square().sum(1)
                )
                output += m.T @ (coefficient[:, None] * (m @ vectors))
        return output

    return dimension, apply


def _top_subspace(dimension: int, rank: int, apply: Any, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    width = min(dimension, rank + 8)
    q = torch.linalg.qr(torch.randn(dimension, width, generator=generator, dtype=DTYPE), mode="reduced").Q
    for _ in range(2):
        q = torch.linalg.qr(apply(q), mode="reduced").Q
    small = q.T @ apply(q)
    eigenvalues, eigenvectors = torch.linalg.eigh((small + small.T) / 2)
    frame = q @ eigenvectors[:, -rank:]
    return torch.linalg.qr(frame, mode="reduced").Q[:, :rank]


def _update_side(
    sources: Sequence[SourceRows],
    assignments: Sequence[torch.Tensor],
    event_energies: Sequence[torch.Tensor],
    eta: float,
    event_count: int,
    unit: int,
    side: str,
    rank: int,
    opposite: torch.Tensor | None,
    incoming: torch.Tensor | None,
    seed: int,
) -> torch.Tensor:
    dimension, apply = _operator(
        sources, assignments, event_energies, eta, event_count, unit, side, opposite
    )
    proposal = _top_subspace(dimension, rank, apply, seed)
    if incoming is None:
        return proposal
    old_capture = float((incoming.T @ apply(incoming)).trace())
    new_capture = float((proposal.T @ apply(proposal)).trace())
    return proposal if new_capture >= old_capture - 1e-10 * max(1.0, abs(old_capture)) else incoming


def _objective_and_reassign(
    sources: Sequence[SourceRows],
    assignments: Sequence[torch.Tensor],
    event_energies: Sequence[torch.Tensor],
    eta: float,
    event_count: int,
    relation: Sequence[torch.Tensor],
    message: Sequence[torch.Tensor],
    *,
    reassign: bool,
) -> tuple[float, tuple[torch.Tensor, ...], int]:
    total = 0.0
    updated = []
    moves = 0
    for source, labels, energy in zip(sources, assignments, event_energies):
        p, m, _alpha, _event, _position, weight = _weights_for_source(
            source, energy, eta, event_count
        )
        norm = p.square().sum(1) * m.square().sum(1)
        captured = torch.stack(
            [
                (p @ u).square().sum(1) * (m @ v).square().sum(1)
                for u, v in zip(relation, message)
            ],
            dim=1,
        )
        proposed = captured.argmax(1).to(torch.long) if reassign else labels
        moves += int((proposed != labels).sum())
        updated.append(proposed)
        chosen = captured.gather(1, proposed[:, None]).squeeze(1)
        total += float((weight * (norm - chosen).clamp_min(0)).sum())
    return total, tuple(updated), moves


def _fit(
    sources: Sequence[SourceRows], *, components: int, relation_rank: int, message_rank: int, seed: int
) -> FitState:
    event_energies, eta, event_count = _event_pair_energies(sources)
    assignments = _initial_assignments(sources, components, seed)
    relation = []
    message = []
    for unit in range(components):
        relation.append(
            _update_side(
                sources, assignments, event_energies, eta, event_count, unit,
                "relation", relation_rank, None, None, seed + 101 * unit + 1,
            )
        )
        message.append(
            _update_side(
                sources, assignments, event_energies, eta, event_count, unit,
                "message", message_rank, None, None, seed + 101 * unit + 2,
            )
        )
    history = []
    converged = False
    for outer in range(30):
        for alternation in range(4):
            for unit in range(components):
                relation[unit] = _update_side(
                    sources, assignments, event_energies, eta, event_count, unit,
                    "relation", relation_rank, message[unit], relation[unit],
                    seed + 10000 * outer + 1000 * alternation + 101 * unit + 11,
                )
                message[unit] = _update_side(
                    sources, assignments, event_energies, eta, event_count, unit,
                    "message", message_rank, relation[unit], message[unit],
                    seed + 10000 * outer + 1000 * alternation + 101 * unit + 12,
                )
        objective, proposed, moves = _objective_and_reassign(
            sources, assignments, event_energies, eta, event_count,
            relation, message, reassign=True,
        )
        counts = [sum(int((labels == unit).sum()) for labels in proposed) for unit in range(components)]
        history.append({"outer": outer, "objective": objective, "label_moves": moves, "pair_counts": counts})
        print(json.dumps({"stage": "pair-fit", "A": components, **history[-1]}), flush=True)
        if min(counts) == 0:
            raise RuntimeError(f"A={components} produced an empty unit.")
        assignments = proposed
        if moves == 0:
            converged = True
            break
    final_objective, replay, moves = _objective_and_reassign(
        sources, assignments, event_energies, eta, event_count,
        relation, message, reassign=True,
    )
    if moves or not converged:
        raise RuntimeError(f"A={components} did not reach assignment closure.")
    history.append({"outer": "replay", "objective": final_objective, "label_moves": moves})
    return FitState(tuple(relation), tuple(message), replay, eta, tuple(history), True)


def _ari(labels: torch.Tensor, truth: torch.Tensor) -> float:
    from unifying_attention.unlearned_projector_selection import adjusted_rand_index

    return adjusted_rand_index(labels.tolist(), truth.tolist())


def _permutation_accuracy(labels: torch.Tensor, truth: torch.Tensor) -> float | None:
    if int(labels.max()) == 0:
        return None
    if int(truth.max()) == 0:
        return None
    direct = float((labels == truth).to(DTYPE).mean())
    swapped = float(((1 - labels) == truth).to(DTYPE).mean())
    return max(direct, swapped)


def _evaluate(
    sources: Sequence[SourceRows],
    state: FitState,
    complement: torch.Tensor,
    *,
    labels: Sequence[torch.Tensor] | None,
    assignment_mode: str,
) -> Mapping[str, Any]:
    pair_num = pair_den = pair_readout_num = message_num = innovation_den = 0.0
    full_num = full_den = dot = pred_norm = 0.0
    event_labels = []
    semantic = []
    subtype = []
    coherences = []
    pair_counts = torch.zeros(len(state.relation), dtype=torch.long)
    energy_counts = torch.zeros(len(state.relation), dtype=DTYPE)
    output_labels = []
    for source_index, source in enumerate(sources):
        p, m, alpha, event, position = _pair_block(source)
        pnorm = p.square().sum(1)
        mnorm = m.square().sum(1)
        capture_relation = torch.stack([(p @ u).square().sum(1) for u in state.relation], dim=1)
        capture_message = torch.stack([(m @ v).square().sum(1) for v in state.message], dim=1)
        if labels is not None:
            local_labels = labels[source_index]
        elif assignment_mode == "joint":
            local_labels = (capture_relation * capture_message).argmax(1)
        elif assignment_mode == "qk":
            local_labels = (capture_relation / pnorm[:, None].clamp_min(1e-30)).argmax(1)
        else:
            raise ValueError("Unknown assignment mode.")
        output_labels.append(local_labels)
        selected_capture = (capture_relation * capture_message).gather(1, local_labels[:, None]).squeeze(1)
        pair_weight = alpha.square()
        pair_num += float((pair_weight * (pnorm * mnorm - selected_capture).clamp_min(0)).sum())
        pair_den += float((pair_weight * pnorm * mnorm).sum())

        predicted_pair = torch.zeros(source.event_count, m.shape[1], dtype=DTYPE)
        predicted_message = torch.zeros_like(predicted_pair)
        exact_innovation = torch.zeros_like(predicted_pair).index_add_(0, event, alpha[:, None] * m)
        energy_table = torch.zeros(source.event_count, len(state.relation), dtype=DTYPE)
        for unit, (u, v) in enumerate(zip(state.relation, state.message)):
            selected = local_labels == unit
            pair_counts[unit] += int(selected.sum())
            local_event = event[selected]
            local_m = m[selected]
            projected = (local_m @ v) @ v.T
            route = ((p[selected] @ u) * u.sum(0)[None, :]).sum(1)
            predicted_pair.index_add_(0, local_event, route[:, None] * projected)
            predicted_message.index_add_(0, local_event, alpha[selected, None] * projected)
            exact_energy = alpha[selected].square() * local_m.square().sum(1)
            energy_table[:, unit].index_add_(0, local_event, exact_energy)
            energy_counts[unit] += exact_energy.sum()
        pair_readout_num += float((exact_innovation - predicted_pair).square().sum())
        message_num += float((exact_innovation - predicted_message).square().sum())
        innovation_den += float(exact_innovation.square().sum())

        dominant = energy_table.argmax(1)
        winning = energy_table.max(1).values / energy_table.sum(1).clamp_min(1e-30)
        event_labels.append(dominant)
        semantic.append(source.semantic_labels)
        subtype.append(source.subtype_labels)
        coherences.append(winning)

        exact_full = source.attention.to(DTYPE) @ source.messages.to(DTYPE)
        nuisance = exact_full - exact_innovation @ complement.T
        predicted_full = nuisance + predicted_pair @ complement.T
        full_num += float((exact_full - predicted_full).square().sum())
        full_den += float(exact_full.square().sum())
        dot += float((exact_full * predicted_full).sum())
        pred_norm += float(predicted_full.square().sum())

    event_labels_t = torch.cat(event_labels)
    semantic_t = torch.cat(semantic)
    subtype_t = torch.cat(subtype)
    total_pair_energy = float(energy_counts.sum())
    return {
        "assignment_mode": assignment_mode,
        "pair_reconstruction_nmse_pooled": pair_num / max(pair_den, 1e-30),
        "W_pair_pooled": pair_readout_num / max(innovation_den, 1e-30),
        "W_msg_pooled": message_num / max(innovation_den, 1e-30),
        "full_write_nmse_pooled": full_num / max(full_den, 1e-30),
        "full_write_cosine_pooled": dot / math.sqrt(max(full_den * pred_norm, 1e-30)),
        "event_coherence_mean": float(torch.cat(coherences).mean()),
        "semantic_event_ari": _ari(event_labels_t, semantic_t),
        "semantic_event_permutation_accuracy": _permutation_accuracy(event_labels_t, semantic_t),
        "subtype_event_ari": _ari(event_labels_t, subtype_t),
        "subtype_event_permutation_accuracy": _permutation_accuracy(event_labels_t, subtype_t),
        "pair_count_by_unit": pair_counts.tolist(),
        "exact_pair_write_energy_fraction_by_unit": (
            energy_counts / max(total_pair_energy, 1e-30)
        ).tolist(),
        "event_count": len(event_labels_t),
        "pair_count": int(pair_counts.sum()),
        "labels": output_labels,
    }


def _strip_labels(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return {key: value for key, value in record.items() if key != "labels"}


def _overlap(frames: Sequence[torch.Tensor]) -> list[list[float]]:
    return [
        [float((left.T @ right).square().sum() / min(left.shape[1], right.shape[1])) for right in frames]
        for left in frames
    ]


def _run_head(
    sources: list[SourceRows],
    *,
    target: str,
    feature_epochs: int,
    feature_lr: float,
    relation_rank: int,
    message_rank: int,
) -> Mapping[str, Any]:
    _model, feature = _fit_feature_map(
        sources, seed=0, epochs=feature_epochs, learning_rate=feature_lr
    )
    _v0, complement, c0 = _fit_common_span(sources)
    if message_rank > complement.shape[1]:
        raise RuntimeError("Requested message rank exceeds the V0 complement.")
    train = [source for source in sources if source.split == "train"]
    holdout = [source for source in sources if source.split == "holdout"]
    rows = []
    for components in (1, 2):
        state = _fit(
            train,
            components=components,
            relation_rank=relation_rank,
            message_rank=message_rank,
            seed=0,
        )
        train_joint = _evaluate(
            train, state, complement, labels=state.assignments, assignment_mode="joint"
        )
        holdout_joint = _evaluate(
            holdout, state, complement, labels=None, assignment_mode="joint"
        )
        holdout_qk = _evaluate(
            holdout, state, complement, labels=None, assignment_mode="qk"
        )
        rows.append(
            {
                "A": components,
                "r": relation_rank,
                "c": message_rank,
                "converged": state.converged,
                "eta_x": state.eta_x,
                "iterations": len([row for row in state.history if isinstance(row.get("outer"), int)]),
                "history": state.history,
                "relation_projector_overlap": _overlap(state.relation),
                "message_projector_overlap": _overlap(state.message),
                "train_joint": _strip_labels(train_joint),
                "holdout_joint": _strip_labels(holdout_joint),
                "holdout_qk_only": _strip_labels(holdout_qk),
                "routing_regret_W_pair": holdout_qk["W_pair_pooled"] - holdout_joint["W_pair_pooled"],
                "routing_regret_W_msg": holdout_qk["W_msg_pooled"] - holdout_joint["W_msg_pooled"],
            }
        )
    return {
        "target": target,
        "train_sources": len(train),
        "holdout_sources": len(holdout),
        "train_events": sum(source.event_count for source in train),
        "holdout_events": sum(source.event_count for source in holdout),
        "message_ambient_rank": int(sources[0].messages.shape[1]),
        "common_V0_rank": c0,
        "message_complement_rank": int(complement.shape[1]),
        "feature_map": feature,
        "fits": rows,
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    _configure_imports(repo_root)
    from unifying_attention.experiments.unlearned_projector_real import (
        load_registered_model_and_tokenizer,
        resolve_registered_snapshot,
    )
    from unifying_attention.unlearned_projector_gate import require_valid_smoke_report

    smoke = Path(args.smoke_report).resolve()
    require_valid_smoke_report(smoke)
    corpus_path = Path(args.corpus).resolve()
    corpus = json.loads(corpus_path.read_text())
    artifacts = resolve_registered_snapshot(allow_download=False)
    if corpus["model"]["revision"] != artifacts.revision:
        raise RuntimeError("Corpus/model revision mismatch.")
    model, tokenizer = load_registered_model_and_tokenizer(artifacts, allow_download=False)
    try:
        l1, l1_audit = _extract_head(
            model, tokenizer, corpus, target="L1H1", layer=1, head=1,
            chunk_size=args.chunk_size, batch_size=args.batch_size,
        )
        l5, l5_audit = _extract_head(
            model, tokenizer, corpus, target="L5H2", layer=5, head=2,
            chunk_size=args.chunk_size, batch_size=args.batch_size,
        )
    finally:
        del model
        del tokenizer
    result = {
        "schema_version": 1,
        "analysis": "algorithm1_learned_rank_one_pair_readout",
        "development_only": True,
        "historical_test_accessed": False,
        "locked_test_accessed": False,
        "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        "smoke_report": str(smoke),
        "configuration": {
            "feature_width": WIDTH,
            "feature_top_k": TOP_K,
            "feature_epochs": args.feature_epochs,
            "feature_learning_rate": args.feature_lr,
            "A": [1, 2],
            "relation_rank": args.relation_rank,
            "message_rank": args.message_rank,
            "restarts": 1,
            "hooi_alternations_per_outer": 4,
            "max_outer_iterations": 30,
            "feature_training_scope": "target-head annotated TRAIN attention rows only",
        },
        "metric_definitions": {
            "pair_reconstruction_nmse_pooled": "sum alpha^2 ||X-Pp X PM||_F^2 / sum alpha^2 ||X||_F^2",
            "W_pair_pooled": "sum_g ||tilde_nu_g-sum_j (1^T Pp_z p_gj) PM_z tilde_mu_gj||^2 / sum_g ||tilde_nu_g||^2",
            "W_msg_pooled": "sum_g ||tilde_nu_g-sum_j alpha_gj PM_z tilde_mu_gj||^2 / sum_g ||tilde_nu_g||^2",
            "full_write_nmse_pooled": "exact BOS/common nuisance plus pair-readout innovation, compared with the complete cached head write",
            "full_write_cosine_pooled": "cosine after stacking all exact and reconstructed complete event writes",
            "joint_assignment": "argmax_a ||U_a^T p||^2 ||V_a^T tilde_mu||^2",
            "qk_only_assignment": "argmax_a ||U_a^T p||^2 / ||p||^2; messages are unread until evaluation",
        },
        "extraction_audit": {"L1H1": l1_audit, "L5H2": l5_audit},
        "L1H1": _run_head(
            l1, target="L1H1", feature_epochs=args.feature_epochs,
            feature_lr=args.feature_lr, relation_rank=args.relation_rank,
            message_rank=args.message_rank,
        ),
        "L5H2": _run_head(
            l5, target="L5H2", feature_epochs=args.feature_epochs,
            feature_lr=args.feature_lr, relation_rank=args.relation_rank,
            message_rank=args.message_rank,
        ),
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--smoke-report", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--feature-epochs", type=int, default=100)
    parser.add_argument("--feature-lr", type=float, default=3e-3)
    parser.add_argument("--relation-rank", type=int, default=32)
    parser.add_argument("--message-rank", type=int, default=64)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if min(
        args.chunk_size, args.batch_size, args.threads, args.feature_epochs,
        args.relation_rank, args.message_rank,
    ) <= 0:
        raise SystemExit("All numerical controls must be positive.")
    torch.set_num_threads(args.threads)
    result = run(args)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
