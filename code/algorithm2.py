"""Algorithm 2: exact joint query-event attention-unit fitting.

This development-only script implements Section 3.3 of updated_note.tex.
It reuses Algorithm 1's authenticated extraction and common-message-span
helpers, but fitting is independent: one hard label is assigned to each active
query event and the fitted objective is R + W_joint.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch

from algorithm1 import (
    SourceRows,
    _ari,
    _configure_imports,
    _extract_head,
    _fit_common_span,
    _kmeans,
    _permutation_accuracy,
)


DTYPE = torch.float64
GRASSMANN_MAX_STEPS = 100
OUTER_MAX_STEPS = 100
MONOTONE_RTOL = 1e-10


@dataclass(frozen=True)
class EventSource:
    base: SourceRows
    q: torch.Tensor
    k: torch.Tensor
    m: torch.Tensor
    alpha: torch.Tensor
    mask: torch.Tensor
    exact_scores: torch.Tensor
    exact_y: torch.Tensor

    @property
    def event_count(self) -> int:
        return int(self.q.shape[0])


@dataclass(frozen=True)
class TrainScales:
    tau_q: float
    tau_k: float
    median_p: float
    median_w: float


@dataclass
class FitState:
    relation: tuple[torch.Tensor, ...]
    message: tuple[torch.Tensor, ...]
    assignments: tuple[torch.Tensor, ...]
    history: tuple[Mapping[str, Any], ...]
    converged: bool


def _orient(frame: torch.Tensor) -> torch.Tensor:
    frame = frame.clone()
    for column in range(frame.shape[1]):
        pivot = int(frame[:, column].abs().argmax())
        if float(frame[pivot, column]) < 0:
            frame[:, column].neg_()
    return frame


def _top_eigenspace(matrix: torch.Tensor, rank: int) -> torch.Tensor:
    symmetric = (matrix + matrix.T) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    order = torch.argsort(eigenvalues, descending=True, stable=True)[:rank]
    return _orient(eigenvectors[:, order].contiguous())


def _qf(frame: torch.Tensor) -> torch.Tensor:
    q, r = torch.linalg.qr(frame, mode="reduced")
    diagonal = torch.diagonal(r)
    signs = torch.where(diagonal < 0, -torch.ones_like(diagonal), torch.ones_like(diagonal))
    return _orient(q * signs[None, :])


def _prepare_sources(sources: Sequence[SourceRows]) -> list[EventSource]:
    prepared = []
    for source in sources:
        if source.complement_messages is None:
            raise RuntimeError("Common message complement has not been fitted.")
        q = source.queries.to(DTYPE)
        k = source.keys.to(DTYPE)
        m = source.complement_messages.to(DTYPE)
        alpha = source.attention.to(DTYPE)
        mask = source.source_mask.to(torch.bool)
        exact_scores = q @ k.T
        exact_scores = torch.where(
            mask, exact_scores, torch.full_like(exact_scores, -torch.inf)
        )
        exact_y = alpha[:, 1:] @ m[1:]
        prepared.append(EventSource(source, q, k, m, alpha, mask, exact_scores, exact_y))
    return prepared


def _fit_scales(sources: Sequence[EventSource]) -> TrainScales:
    event_count = sum(source.event_count for source in sources)
    tau_q = sum(float(source.q.square().sum()) for source in sources) / event_count
    tau_k_total = 0.0
    p_energies = []
    w_energies = []
    for source in sources:
        nonbos_mass = source.alpha[:, 1:].sum(1).clamp_min(1e-30)
        key_norm = source.k[1:].square().sum(1)
        tau_k_total += float(
            ((source.alpha[:, 1:] @ key_norm) / nonbos_mass).sum()
        )
        message_norm = source.m[1:].square().sum(1)
        p_energies.append(source.alpha[:, 1:].square() @ message_norm)
        w_energies.append(source.exact_y.square().sum(1))
    tau_k = tau_k_total / event_count
    p = torch.cat(p_energies)
    w = torch.cat(w_energies)
    p_positive = p[p > 0]
    w_positive = w[w > 0]
    if not len(p_positive) or not len(w_positive):
        raise RuntimeError("TRAIN loss scales have no positive median.")
    return TrainScales(
        tau_q=tau_q,
        tau_k=tau_k,
        median_p=float(p_positive.median()),
        median_w=float(w_positive.median()),
    )


def _denominators(source: EventSource, scales: TrainScales) -> tuple[torch.Tensor, torch.Tensor]:
    message_norm = source.m[1:].square().sum(1)
    raw_p = source.alpha[:, 1:].square() @ message_norm
    raw_w = source.exact_y.square().sum(1)
    d_p = raw_p + 1e-8 * scales.median_p
    d_w = torch.maximum(raw_w, torch.full_like(raw_w, 1e-2 * scales.median_w))
    return d_p, d_w


def _predicted_attention(source: EventSource, frame: torch.Tensor) -> torch.Tensor:
    projected = (source.q @ frame) @ (source.k @ frame).T
    projected = torch.where(
        source.mask, projected, torch.full_like(projected, -torch.inf)
    )
    logits = torch.cat((source.exact_scores[:, :1], projected[:, 1:]), dim=1)
    return torch.softmax(logits, dim=1)


def _unit_terms(
    source: EventSource,
    event_indices: torch.Tensor,
    relation: torch.Tensor,
    message: torch.Tensor,
    scales: TrainScales,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q = source.q.index_select(0, event_indices)
    alpha = source.alpha.index_select(0, event_indices)
    q_residual = q.square().sum(1) - (q @ relation).square().sum(1)
    key_residual = source.k[1:].square().sum(1) - (source.k[1:] @ relation).square().sum(1)
    nonbos_mass = alpha[:, 1:].sum(1).clamp_min(1e-30)
    r = 0.5 * (
        q_residual / scales.tau_q
        + (alpha[:, 1:] @ key_residual) / (scales.tau_k * nonbos_mass)
    )
    predicted_alpha = _predicted_attention(source, relation).index_select(0, event_indices)
    b = predicted_alpha[:, 1:] @ source.m[1:]
    predicted_y = (b @ message) @ message.T
    _d_p, d_w = _denominators(source, scales)
    y = source.exact_y.index_select(0, event_indices)
    w = (y - predicted_y).square().sum(1) / d_w.index_select(0, event_indices)
    return r, w, predicted_alpha, b


def _objective(
    sources: Sequence[EventSource],
    assignments: Sequence[torch.Tensor],
    relation: Sequence[torch.Tensor],
    message: Sequence[torch.Tensor],
    scales: TrainScales,
) -> torch.Tensor:
    total = torch.zeros((), dtype=DTYPE)
    event_count = sum(source.event_count for source in sources)
    for source, labels in zip(sources, assignments):
        for unit, (u, v) in enumerate(zip(relation, message)):
            indices = (labels == unit).nonzero(as_tuple=True)[0]
            if not len(indices):
                continue
            r, w, _alpha, _b = _unit_terms(source, indices, u, v, scales)
            total = total + r.sum() + w.sum()
    return total / event_count


def _relation_costs(
    source: EventSource,
    relation: Sequence[torch.Tensor],
    scales: TrainScales,
) -> torch.Tensor:
    rows = []
    for u in relation:
        q_residual = source.q.square().sum(1) - (source.q @ u).square().sum(1)
        key_residual = source.k[1:].square().sum(1) - (source.k[1:] @ u).square().sum(1)
        nonbos_mass = source.alpha[:, 1:].sum(1).clamp_min(1e-30)
        rows.append(
            0.5
            * (
                q_residual / scales.tau_q
                + (source.alpha[:, 1:] @ key_residual)
                / (scales.tau_k * nonbos_mass)
            )
        )
    return torch.stack(rows, dim=1)


def _event_costs(
    sources: Sequence[EventSource],
    relation: Sequence[torch.Tensor],
    message: Sequence[torch.Tensor],
    scales: TrainScales,
    *,
    qk_only: bool,
) -> tuple[tuple[torch.Tensor, ...], float]:
    labels = []
    total = 0.0
    event_count = sum(source.event_count for source in sources)
    with torch.no_grad():
        for source in sources:
            costs = _relation_costs(source, relation, scales)
            if not qk_only:
                _d_p, d_w = _denominators(source, scales)
                for unit, (u, v) in enumerate(zip(relation, message)):
                    alpha_hat = _predicted_attention(source, u)
                    b = alpha_hat[:, 1:] @ source.m[1:]
                    y_hat = (b @ v) @ v.T
                    costs[:, unit] += (
                        (source.exact_y - y_hat).square().sum(1) / d_w
                    )
            selected = costs.argmin(1)
            labels.append(selected)
            total += float(costs.gather(1, selected[:, None]).sum())
    return tuple(labels), total / event_count


def _event_coordinates(sources: Sequence[EventSource]) -> tuple[torch.Tensor, list[int]]:
    queries = torch.cat([source.q for source in sources], dim=0)
    key_means = torch.cat(
        [
            (source.alpha[:, 1:] @ source.k[1:])
            / source.alpha[:, 1:].sum(1, keepdim=True).clamp_min(1e-30)
            for source in sources
        ],
        dim=0,
    )
    blocks = []
    for values in (queries, key_means):
        centered = values - values.mean(0, keepdim=True)
        covariance = centered.T @ centered
        basis = _top_eigenspace(covariance, 3)
        blocks.append(centered @ basis)
    coordinates = torch.cat(blocks, dim=1)
    rms = coordinates.square().mean(0).sqrt().clamp_min(torch.finfo(DTYPE).eps)
    return (coordinates / rms).to(torch.float32), [source.event_count for source in sources]


def _initial_assignments(
    sources: Sequence[EventSource], components: int, seed: int
) -> tuple[torch.Tensor, ...]:
    coordinates, lengths = _event_coordinates(sources)
    labels = _kmeans(coordinates, components, seed)
    pieces = tuple(labels.split(lengths))
    counts = [sum(int((piece == unit).sum()) for piece in pieces) for unit in range(components)]
    if min(counts) == 0:
        raise RuntimeError("Relation-only initialization produced an empty unit.")
    return pieces


def _spectral_relation_update(
    sources: Sequence[EventSource],
    assignments: Sequence[torch.Tensor],
    components: int,
    rank: int,
    scales: TrainScales,
) -> tuple[torch.Tensor, ...]:
    event_count = sum(source.event_count for source in sources)
    matrices = [torch.zeros(256, 256, dtype=DTYPE) for _ in range(components)]
    for source, labels in zip(sources, assignments):
        key = source.k[1:]
        for unit in range(components):
            indices = (labels == unit).nonzero(as_tuple=True)[0]
            if not len(indices):
                continue
            q = source.q.index_select(0, indices)
            alpha = source.alpha.index_select(0, indices)[:, 1:]
            mass = alpha.sum(1, keepdim=True).clamp_min(1e-30)
            key_weight = (alpha / mass).sum(0)
            matrices[unit] += (
                q.T @ q / scales.tau_q
                + key.T @ (key_weight[:, None] * key) / scales.tau_k
            ) / (2 * event_count)
    return tuple(_top_eigenspace(matrix, rank) for matrix in matrices)


def _accept_spectral_relation(
    sources: Sequence[EventSource],
    assignments: Sequence[torch.Tensor],
    relation: Sequence[torch.Tensor],
    message: Sequence[torch.Tensor],
    proposals: Sequence[torch.Tensor],
    scales: TrainScales,
) -> tuple[tuple[torch.Tensor, ...], float]:
    current = tuple(frame.clone() for frame in relation)
    current_loss = float(_objective(sources, assignments, current, message, scales))
    for unit, proposal in enumerate(proposals):
        trial = list(current)
        trial[unit] = proposal
        trial_loss = float(_objective(sources, assignments, trial, message, scales))
        tolerance = MONOTONE_RTOL * max(1.0, abs(current_loss))
        if math.isfinite(trial_loss) and trial_loss <= current_loss + tolerance:
            current = tuple(trial)
            current_loss = min(current_loss, trial_loss)
    return current, current_loss


def _grassmann_refine_relation(
    sources: Sequence[EventSource],
    assignments: Sequence[torch.Tensor],
    relation: Sequence[torch.Tensor],
    message: Sequence[torch.Tensor],
    scales: TrainScales,
) -> tuple[tuple[torch.Tensor, ...], float, Mapping[str, Any]]:
    frames = tuple(frame.detach().clone() for frame in relation)
    current_loss = float(_objective(sources, assignments, frames, message, scales))
    delta = 1e-2
    accepted = 0
    small_improvements = 0
    reason = "iteration_cap"
    for iteration in range(1, GRASSMANN_MAX_STEPS + 1):
        variables = tuple(frame.detach().clone().requires_grad_(True) for frame in frames)
        loss = _objective(sources, assignments, variables, message, scales)
        gradients = torch.autograd.grad(loss, variables)
        tangents = tuple(
            gradient - frame @ (frame.T @ gradient)
            for frame, gradient in zip(variables, gradients)
        )
        tangent_norm = math.sqrt(sum(float(t.square().sum()) for t in tangents))
        if tangent_norm <= 1e-10:
            reason = "tangent"
            break
        accepted_step = False
        while delta >= 1e-8:
            trials = tuple(
                _qf(
                    frame.detach()
                    - delta
                    * math.sqrt(frame.shape[1])
                    * tangent.detach()
                    / max(float(tangent.norm()), 1e-30)
                )
                for frame, tangent in zip(variables, tangents)
            )
            if any(
                not torch.isfinite(trial).all()
                or float((trial.T @ trial - torch.eye(trial.shape[1], dtype=DTYPE)).norm())
                > 1e-8
                for trial in trials
            ):
                delta /= 2
                continue
            trial_loss = float(_objective(sources, assignments, trials, message, scales))
            armijo = current_loss + 1e-4 * sum(
                float((tangent.detach() * (trial - frame.detach())).sum())
                for tangent, trial, frame in zip(tangents, trials, variables)
            )
            tolerance = MONOTONE_RTOL * max(1.0, abs(current_loss))
            if (
                math.isfinite(trial_loss)
                and trial_loss <= armijo + tolerance
                and trial_loss <= current_loss + tolerance
            ):
                improvement = current_loss - trial_loss
                frames = trials
                current_loss = min(current_loss, trial_loss)
                accepted += 1
                accepted_step = True
                if improvement <= 1e-7 * max(1.0, abs(current_loss)):
                    small_improvements += 1
                else:
                    small_improvements = 0
                delta = min(1.2 * delta, 5e-2)
                break
            delta /= 2
        if not accepted_step:
            reason = "step_floor"
            break
        if small_improvements >= 3:
            reason = "small_improvement"
            break
    return frames, current_loss, {
        "accepted_steps": accepted,
        "stop_reason": reason,
        "final_step_size": delta,
    }


def _exact_message_update(
    sources: Sequence[EventSource],
    assignments: Sequence[torch.Tensor],
    relation: Sequence[torch.Tensor],
    message: Sequence[torch.Tensor] | None,
    components: int,
    rank: int,
    scales: TrainScales,
) -> tuple[tuple[torch.Tensor, ...], float]:
    event_count = sum(source.event_count for source in sources)
    dimension = int(sources[0].m.shape[1])
    matrices = [torch.zeros(dimension, dimension, dtype=DTYPE) for _ in range(components)]
    with torch.no_grad():
        for source, labels in zip(sources, assignments):
            _d_p, d_w = _denominators(source, scales)
            for unit, u in enumerate(relation):
                indices = (labels == unit).nonzero(as_tuple=True)[0]
                if not len(indices):
                    continue
                alpha_hat = _predicted_attention(source, u).index_select(0, indices)
                b = alpha_hat[:, 1:] @ source.m[1:]
                y = source.exact_y.index_select(0, indices)
                weight = 1.0 / (event_count * d_w.index_select(0, indices))
                matrices[unit] += (
                    y.T @ (weight[:, None] * b)
                    + b.T @ (weight[:, None] * y)
                    - b.T @ (weight[:, None] * b)
                )
    proposals = tuple(_top_eigenspace(matrix, rank) for matrix in matrices)
    if message is None:
        loss = float(_objective(sources, assignments, relation, proposals, scales))
        return proposals, loss
    current = tuple(frame.clone() for frame in message)
    current_loss = float(_objective(sources, assignments, relation, current, scales))
    for unit, proposal in enumerate(proposals):
        trial = list(current)
        trial[unit] = proposal
        trial_loss = float(_objective(sources, assignments, relation, trial, scales))
        tolerance = MONOTONE_RTOL * max(1.0, abs(current_loss))
        if not math.isfinite(trial_loss) or trial_loss > current_loss + tolerance:
            continue
        current = tuple(trial)
        current_loss = min(current_loss, trial_loss)
    return current, current_loss


def _fit(
    sources: Sequence[EventSource],
    *,
    components: int,
    relation_rank: int,
    message_rank: int,
    seed: int,
    scales: TrainScales,
) -> FitState:
    assignments = _initial_assignments(sources, components, seed)
    relation = _spectral_relation_update(
        sources, assignments, components, relation_rank, scales
    )
    message, current_loss = _exact_message_update(
        sources, assignments, relation, None, components, message_rank, scales
    )
    history: list[Mapping[str, Any]] = []
    converged = False
    for outer in range(OUTER_MAX_STEPS):
        outer_start = current_loss
        proposals = _spectral_relation_update(
            sources, assignments, components, relation_rank, scales
        )
        relation, spectral_loss = _accept_spectral_relation(
            sources, assignments, relation, message, proposals, scales
        )
        relation, relation_loss, refinement = _grassmann_refine_relation(
            sources, assignments, relation, message, scales
        )
        if relation_loss > outer_start + MONOTONE_RTOL * max(1.0, abs(outer_start)):
            raise RuntimeError("Relation M-step increased the objective.")
        message, message_loss = _exact_message_update(
            sources, assignments, relation, message,
            components, message_rank, scales,
        )
        if message_loss > relation_loss + MONOTONE_RTOL * max(1.0, abs(relation_loss)):
            raise RuntimeError("Message M-step increased the objective.")
        proposed, assignment_loss = _event_costs(
            sources, relation, message, scales, qk_only=False
        )
        counts = [
            sum(int((labels == unit).sum()) for labels in proposed)
            for unit in range(components)
        ]
        if min(counts) == 0:
            raise RuntimeError(f"A={components} E-step produced an empty unit.")
        moves = sum(
            int((old != new).sum()) for old, new in zip(assignments, proposed)
        )
        if assignment_loss > message_loss + MONOTONE_RTOL * max(1.0, abs(message_loss)):
            raise RuntimeError("E-step increased the objective.")
        relative_decrease = (outer_start - assignment_loss) / max(1.0, abs(outer_start))
        record = {
            "outer": outer,
            "start_objective": outer_start,
            "post_spectral_objective": spectral_loss,
            "post_relation_objective": relation_loss,
            "post_message_objective": message_loss,
            "post_assignment_objective": assignment_loss,
            "relative_decrease": relative_decrease,
            "label_moves": moves,
            "event_counts": counts,
            "relation_refinement": refinement,
        }
        history.append(record)
        print(json.dumps({"stage": "event-fit", "A": components, **record}), flush=True)
        assignments = proposed
        current_loss = assignment_loss
        if moves == 0 and relative_decrease <= 1e-6:
            converged = True
            break
    if not converged:
        raise RuntimeError(f"A={components} hit the outer-iteration cap.")
    replay, replay_loss = _event_costs(
        sources, relation, message, scales, qk_only=False
    )
    replay_moves = sum(
        int((old != new).sum()) for old, new in zip(assignments, replay)
    )
    if replay_moves:
        raise RuntimeError(f"A={components} failed zero-move closure.")
    history.append(
        {"outer": "closure", "objective": replay_loss, "label_moves": replay_moves}
    )
    return FitState(tuple(relation), tuple(message), replay, tuple(history), True)


def _overlap(frames: Sequence[torch.Tensor]) -> list[list[float]]:
    return [
        [
            float((left.T @ right).square().sum() / min(left.shape[1], right.shape[1]))
            for right in frames
        ]
        for left in frames
    ]


def _metric_record(
    sources: Sequence[EventSource],
    assignments: Sequence[torch.Tensor],
    relation: Sequence[torch.Tensor],
    message: Sequence[torch.Tensor],
    scales: TrainScales,
    v0: torch.Tensor,
    complement: torch.Tensor,
    *,
    assignment_mode: str,
) -> Mapping[str, Any]:
    event_count = sum(source.event_count for source in sources)
    r_sum = p_macro = p_num = p_den = 0.0
    w_joint_macro = w_joint_num = 0.0
    w_msg_macro = w_msg_num = 0.0
    w_rel_macro = w_rel_num = 0.0
    innovation_den = 0.0
    full_num = full_den = full_dot = full_pred_norm = 0.0
    kl_sum = bos_error = nonbos_mass_error = 0.0
    calibration_cross = calibration_exact = calibration_projected = 0.0
    floor_count = 0
    occupancy = torch.zeros(len(relation), dtype=DTYPE)
    energy = torch.zeros(len(relation), dtype=DTYPE)
    semantic = []
    subtype = []
    discovered = []
    example_candidates: list[list[Mapping[str, Any]]] = [
        [] for _ in relation
    ]
    with torch.no_grad():
        for source, labels in zip(sources, assignments):
            d_p, d_w = _denominators(source, scales)
            raw_w = source.exact_y.square().sum(1)
            floor_count += int((raw_w < 1e-2 * scales.median_w).sum())
            exact_full = source.alpha @ source.base.messages.to(DTYPE)
            predicted_full = torch.zeros_like(exact_full)
            predicted_innovation = torch.zeros_like(source.exact_y)
            predicted_message = torch.zeros_like(source.exact_y)
            predicted_relation = torch.zeros_like(source.exact_y)
            predicted_alpha_all = torch.zeros_like(source.alpha)
            projected_logits_all = torch.zeros_like(source.exact_scores)
            r_local = _relation_costs(source, relation, scales)
            message_norm = source.m[1:].square().sum(1)
            for unit, (u, v) in enumerate(zip(relation, message)):
                indices = (labels == unit).nonzero(as_tuple=True)[0]
                if not len(indices):
                    continue
                occupancy[unit] += len(indices)
                alpha_exact = source.alpha.index_select(0, indices)
                alpha_hat = _predicted_attention(source, u).index_select(0, indices)
                predicted_alpha_all.index_copy_(0, indices, alpha_hat)
                projected = (source.q.index_select(0, indices) @ u) @ (source.k @ u).T
                projected_logits_all.index_copy_(0, indices, projected)
                b = alpha_hat[:, 1:] @ source.m[1:]
                y = source.exact_y.index_select(0, indices)
                y_joint = (b @ v) @ v.T
                y_msg = (y @ v) @ v.T
                predicted_innovation.index_copy_(0, indices, y_joint)
                predicted_message.index_copy_(0, indices, y_msg)
                predicted_relation.index_copy_(0, indices, b)
                residual_message = message_norm - (source.m[1:] @ v).square().sum(1)
                p_event = alpha_exact[:, 1:].square() @ residual_message
                p_macro += float((p_event / d_p.index_select(0, indices)).sum())
                p_num += float(p_event.sum())
                p_den += float(
                    (alpha_exact[:, 1:].square() @ message_norm).sum()
                )
                event_energy = alpha_exact[:, 1:].square() @ message_norm
                energy[unit] += float(event_energy.sum())

                original = source.base.messages.to(DTYPE)
                common_coordinates = original @ v0
                common = (alpha_hat[:, 1:] @ common_coordinates[1:]) @ v0.T
                bos = alpha_hat[:, :1] * original[:1]
                full = bos + common + y_joint @ complement.T
                predicted_full.index_copy_(0, indices, full)

                for local_row, event_index in enumerate(indices.tolist()):
                    best_source = int(alpha_exact[local_row, 1:].argmax()) + 1
                    example_candidates[unit].append(
                        {
                            "source_id": source.base.source_id,
                            "family": source.base.families[event_index],
                            "query_position": int(source.base.query_positions[event_index]),
                            "query_token": source.base.tokens[
                                int(source.base.query_positions[event_index])
                            ],
                            "source_position": best_source,
                            "source_token": source.base.tokens[best_source],
                            "attention": float(alpha_exact[local_row, best_source]),
                            "innovation_energy": float(y[local_row].square().sum()),
                        }
                    )

            selected_r = r_local.gather(1, labels[:, None]).squeeze(1)
            r_sum += float(selected_r.sum())
            joint_error = (source.exact_y - predicted_innovation).square().sum(1)
            msg_error = (source.exact_y - predicted_message).square().sum(1)
            rel_error = (source.exact_y - predicted_relation).square().sum(1)
            w_joint_macro += float((joint_error / d_w).sum())
            w_msg_macro += float((msg_error / d_w).sum())
            w_rel_macro += float((rel_error / d_w).sum())
            w_joint_num += float(joint_error.sum())
            w_msg_num += float(msg_error.sum())
            w_rel_num += float(rel_error.sum())
            innovation_den += float(source.exact_y.square().sum())
            full_error = (exact_full - predicted_full).square().sum()
            full_num += float(full_error)
            full_den += float(exact_full.square().sum())
            full_dot += float((exact_full * predicted_full).sum())
            full_pred_norm += float(predicted_full.square().sum())

            positive = source.alpha > 0
            kl_sum += float(
                (
                    source.alpha[positive]
                    * (
                        source.alpha[positive].clamp_min(1e-30).log()
                        - predicted_alpha_all[positive].clamp_min(1e-30).log()
                    )
                ).sum()
            )
            bos_error += float(
                (source.alpha[:, 0] - predicted_alpha_all[:, 0]).abs().sum()
            )
            nonbos_mass_error += float(
                (
                    source.alpha[:, 1:].sum(1)
                    - predicted_alpha_all[:, 1:].sum(1)
                ).abs().sum()
            )
            weight = source.alpha[:, 1:].square()
            weight_sum = weight.sum(1, keepdim=True).clamp_min(1e-30)
            valid_nonbos = source.mask[:, 1:]
            exact = torch.where(
                valid_nonbos, source.exact_scores[:, 1:], torch.zeros_like(weight)
            )
            projected = torch.where(
                valid_nonbos, projected_logits_all[:, 1:], torch.zeros_like(weight)
            )
            exact_centered = exact - (weight * exact).sum(1, keepdim=True) / weight_sum
            projected_centered = (
                projected - (weight * projected).sum(1, keepdim=True) / weight_sum
            )
            calibration_cross += float((weight * exact_centered * projected_centered).sum())
            calibration_exact += float((weight * exact_centered.square()).sum())
            calibration_projected += float((weight * projected_centered.square()).sum())
            semantic.append(source.base.semantic_labels)
            subtype.append(source.base.subtype_labels)
            discovered.append(labels)

    discovered_t = torch.cat(discovered)
    semantic_t = torch.cat(semantic)
    subtype_t = torch.cat(subtype)
    total_energy = float(energy.sum())
    examples = []
    for rows in example_candidates:
        examples.append(
            sorted(rows, key=lambda row: (-row["innovation_energy"], row["source_id"]))[:3]
        )
    return {
        "assignment_mode": assignment_mode,
        "R_macro": r_sum / event_count,
        "P_macro": p_macro / event_count,
        "P_pooled": p_num / max(p_den, 1e-30),
        "W_joint_macro": w_joint_macro / event_count,
        "W_joint_pooled": w_joint_num / max(innovation_den, 1e-30),
        "W_msg_macro": w_msg_macro / event_count,
        "W_msg_pooled": w_msg_num / max(innovation_den, 1e-30),
        "W_rel_macro": w_rel_macro / event_count,
        "W_rel_pooled": w_rel_num / max(innovation_den, 1e-30),
        "full_write_nmse_pooled": full_num / max(full_den, 1e-30),
        "full_write_cosine_pooled": full_dot
        / math.sqrt(max(full_den * full_pred_norm, 1e-30)),
        "attention_kl_mean": kl_sum / event_count,
        "bos_attention_absolute_error_mean": bos_error / event_count,
        "nonbos_mass_absolute_error_mean": nonbos_mass_error / event_count,
        "score_calibration_slope": calibration_cross / max(calibration_exact, 1e-30),
        "score_sd_ratio": math.sqrt(
            calibration_projected / max(calibration_exact, 1e-30)
        ),
        "W_floor_fraction": floor_count / event_count,
        "event_occupancy_fraction_by_unit": (occupancy / event_count).tolist(),
        "exact_pair_write_energy_fraction_by_unit": (
            energy / max(total_energy, 1e-30)
        ).tolist(),
        "semantic_event_ari": _ari(discovered_t, semantic_t),
        "semantic_event_permutation_accuracy": _permutation_accuracy(
            discovered_t, semantic_t
        ),
        "subtype_event_ari": _ari(discovered_t, subtype_t),
        "subtype_event_permutation_accuracy": _permutation_accuracy(
            discovered_t, subtype_t
        ),
        "event_count": event_count,
        "examples_by_unit": examples,
    }


def _run_head(
    raw_sources: list[SourceRows],
    *,
    target: str,
    relation_rank: int,
    message_rank: int,
) -> Mapping[str, Any]:
    v0, complement, c0 = _fit_common_span(raw_sources)
    if message_rank > complement.shape[1]:
        raise RuntimeError("Requested message rank exceeds the V0 complement.")
    prepared = _prepare_sources(raw_sources)
    train = [source for source in prepared if source.base.split == "train"]
    holdout = [source for source in prepared if source.base.split == "holdout"]
    scales = _fit_scales(train)
    rows = []
    for components in (1, 2):
        state = _fit(
            train,
            components=components,
            relation_rank=relation_rank,
            message_rank=message_rank,
            seed=0,
            scales=scales,
        )
        train_metrics = _metric_record(
            train, state.assignments, state.relation, state.message,
            scales, v0, complement, assignment_mode="joint",
        )
        holdout_joint_labels, _ = _event_costs(
            holdout, state.relation, state.message, scales, qk_only=False
        )
        holdout_qk_labels, _ = _event_costs(
            holdout, state.relation, state.message, scales, qk_only=True
        )
        holdout_joint = _metric_record(
            holdout, holdout_joint_labels, state.relation, state.message,
            scales, v0, complement, assignment_mode="joint",
        )
        holdout_qk = _metric_record(
            holdout, holdout_qk_labels, state.relation, state.message,
            scales, v0, complement, assignment_mode="qk_only",
        )
        rows.append(
            {
                "A": components,
                "r": relation_rank,
                "c": message_rank,
                "converged": state.converged,
                "iterations": len(
                    [row for row in state.history if isinstance(row.get("outer"), int)]
                ),
                "history": state.history,
                "relation_projector_overlap": _overlap(state.relation),
                "message_projector_overlap": _overlap(state.message),
                "train_joint": train_metrics,
                "holdout_joint": holdout_joint,
                "holdout_qk_only": holdout_qk,
                "routing_regret_W_joint_pooled": (
                    holdout_qk["W_joint_pooled"]
                    - holdout_joint["W_joint_pooled"]
                ),
            }
        )
    return {
        "target": target,
        "train_sources": len(train),
        "holdout_sources": len(holdout),
        "train_events": sum(source.event_count for source in train),
        "holdout_events": sum(source.event_count for source in holdout),
        "common_V0_rank": c0,
        "message_ambient_rank": int(raw_sources[0].messages.shape[1]),
        "message_complement_rank": int(complement.shape[1]),
        "train_scales": {
            "tau_q": scales.tau_q,
            "tau_k": scales.tau_k,
            "median_P_energy": scales.median_p,
            "median_W_energy": scales.median_w,
        },
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
    dependency = Path(__file__).with_name("algorithm1.py")
    return {
        "schema_version": 1,
        "analysis": "algorithm2_exact_joint_query_event",
        "development_only": True,
        "historical_test_accessed": False,
        "locked_test_accessed": False,
        "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        "smoke_report": str(smoke),
        "configuration": {
            "A": [1, 2],
            "relation_rank": args.relation_rank,
            "message_rank": args.message_rank,
            "restarts": 1,
            "seed": 0,
            "max_outer_iterations": OUTER_MAX_STEPS,
            "max_grassmann_steps": GRASSMANN_MAX_STEPS,
            "initialization": "relation-only event Q plus attended-key PCA/k-means++",
            "fit_objective": "mean_g [R_g,z + W_joint_g(z)]",
            "event_label": "one hard label per complete active non-BOS query row",
        },
        "metric_definitions": {
            "R_macro": "mean event affine-free Q/K subspace residual R_g,z",
            "P_macro": "mean_g sum_j alpha_gj^2 ||tilde_mu_gj-PM_z tilde_mu_gj||^2 / D_P,g",
            "P_pooled": "pooled source-message squared projection error divided by pooled alpha^2 message energy",
            "W_joint": "projected Q/K is re-softmaxed with exact BOS and the routed projected message write is compared with tilde_nu",
            "W_msg": "exact attention with routed projected messages; isolates message compression",
            "W_rel": "reconstructed attention with exact complement messages; isolates relation-side write error",
            "macro": "equal-event mean using the frozen TRAIN denominator D_W,g",
            "pooled": "sum squared error over events divided by sum ||tilde_nu_g||^2",
            "full_write_nmse_pooled": "complete predicted BOS plus common V0 plus unit innovation write error divided by complete write energy",
            "full_write_cosine_pooled": "cosine after stacking complete exact and predicted event writes",
            "attention_kl_mean": "mean event KL(alpha_exact || alpha_reconstructed) on the complete causal row",
            "joint_assignment": "argmin_a R_g,a + W_joint_g(a) with frozen parameters",
            "qk_only_assignment": "argmin_a R_g,a without reading messages or V_a",
            "zero_move_closure": "the final post-M complete E-step changes no TRAIN event labels",
            "monotonicity": "every accepted M and E block is non-increasing within 1e-10 max(1,|L|)",
        },
        "extraction_dependency_sha256": hashlib.sha256(dependency.read_bytes()).hexdigest(),
        "extraction_audit": {"L1H1": l1_audit, "L5H2": l5_audit},
        "L1H1": _run_head(
            l1, target="L1H1",
            relation_rank=args.relation_rank, message_rank=args.message_rank,
        ),
        "L5H2": _run_head(
            l5, target="L5H2",
            relation_rank=args.relation_rank, message_rank=args.message_rank,
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--smoke-report", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--relation-rank", type=int, default=32)
    parser.add_argument("--message-rank", type=int, default=64)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if min(
        args.chunk_size, args.batch_size, args.threads,
        args.relation_rank, args.message_rank,
    ) <= 0:
        raise SystemExit("All numerical controls must be positive.")
    torch.set_num_threads(args.threads)
    result = run(args)
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
