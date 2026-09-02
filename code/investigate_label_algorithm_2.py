"""Investigate label alignment for Algorithm 2's event-level estimator.

This development-only diagnostic preserves Algorithm 2's r=32, c=64,
R+W_joint M/E updates.  It audits the exact initialization geometry, traces
behavioral alignment during fitting, and compares K-means with an oracle
L1H1 word/newline initialization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

import algorithm2 as alg
from algorithm1 import SourceRows, _ari, _configure_imports, _extract_head, _fit_common_span, _kmeans


DTYPE = torch.float64


def _truth(source: alg.EventSource, target: str) -> torch.Tensor:
    # L1H1 has a genuine proposed word-like/newline split.  L5H2's labels are
    # strict/tokenization-tolerant subtypes of one suspected induction unit.
    return source.base.semantic_labels if target == "L1H1" else source.base.subtype_labels


def _alignment(labels: Sequence[torch.Tensor], sources: Sequence[alg.EventSource], target: str) -> Mapping[str, Any]:
    predicted = torch.cat(tuple(labels)).to(torch.long)
    truth = torch.cat([_truth(source, target) for source in sources]).to(torch.long)
    direct = float((predicted == truth).to(DTYPE).mean())
    swapped = float(((1 - predicted) == truth).to(DTYPE).mean())
    aligned = predicted if direct >= swapped else 1 - predicted
    recalls = []
    for value in (0, 1):
        selected = truth == value
        recalls.append(float((aligned[selected] == value).to(DTYPE).mean()))
    counts = torch.bincount(truth, minlength=2).to(DTYPE)
    return {
        "ari": alg._ari(predicted, truth),
        "permutation_accuracy": max(direct, swapped),
        "balanced_accuracy": sum(recalls) / 2,
        "recall_by_truth_class": recalls,
        "truth_counts": counts.to(torch.long).tolist(),
        "truth_fraction": (counts / counts.sum()).tolist(),
        "predicted_fraction": (torch.bincount(predicted, minlength=2).to(DTYPE) / len(predicted)).tolist(),
    }


def _fit_coordinate_model(train: Sequence[alg.EventSource]) -> Mapping[str, torch.Tensor]:
    queries = torch.cat([source.q for source in train])
    key_means = torch.cat([
        (source.alpha[:, 1:] @ source.k[1:])
        / source.alpha[:, 1:].sum(1, keepdim=True).clamp_min(1e-30)
        for source in train
    ])
    model: dict[str, torch.Tensor] = {}
    blocks = []
    for name, values in (("q", queries), ("k", key_means)):
        mean = values.mean(0, keepdim=True)
        centered = values - mean
        basis = alg._top_eigenspace(centered.T @ centered, 3)
        model[f"{name}_mean"] = mean
        model[f"{name}_basis"] = basis
        blocks.append(centered @ basis)
    raw = torch.cat(blocks, dim=1)
    model["rms"] = raw.square().mean(0).sqrt().clamp_min(torch.finfo(DTYPE).eps)
    return model


def _coordinates(sources: Sequence[alg.EventSource], model: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, list[int]]:
    rows = []
    lengths = []
    for source in sources:
        key_mean = (
            (source.alpha[:, 1:] @ source.k[1:])
            / source.alpha[:, 1:].sum(1, keepdim=True).clamp_min(1e-30)
        )
        q = (source.q - model["q_mean"]) @ model["q_basis"]
        k = (key_mean - model["k_mean"]) @ model["k_basis"]
        rows.append(torch.cat((q, k), dim=1) / model["rms"])
        lengths.append(source.event_count)
    return torch.cat(rows).to(torch.float32), lengths


def _split(labels: torch.Tensor, lengths: Sequence[int]) -> tuple[torch.Tensor, ...]:
    return tuple(labels.split(list(lengths)))


def _geometry(train: Sequence[alg.EventSource], holdout: Sequence[alg.EventSource], target: str) -> Mapping[str, Any]:
    model = _fit_coordinate_model(train)
    train_x, train_lengths = _coordinates(train, model)
    holdout_x, holdout_lengths = _coordinates(holdout, model)

    kmeans = _kmeans(train_x, 2, 0)
    centers = torch.stack([train_x[kmeans == unit].mean(0) for unit in range(2)])
    holdout_kmeans = torch.cdist(holdout_x, centers).argmin(1)

    train_truth = torch.cat([_truth(source, target) for source in train])
    behavior_centers = torch.stack([train_x[train_truth == unit].mean(0) for unit in range(2)])
    supervised_train = torch.cdist(train_x, behavior_centers).argmin(1)
    supervised_holdout = torch.cdist(holdout_x, behavior_centers).argmin(1)
    return {
        "coordinates": "top-3 TRAIN PCA coordinates of Q plus top-3 TRAIN PCA coordinates of the exact-attention-weighted non-BOS key mean; each coordinate divided by its TRAIN RMS",
        "kmeans_train": _alignment(_split(kmeans, train_lengths), train, target),
        "nearest_frozen_kmeans_center_holdout": _alignment(_split(holdout_kmeans, holdout_lengths), holdout, target),
        "label_informed_nearest_behavior_centroid": {
            "train": _alignment(_split(supervised_train, train_lengths), train, target),
            "holdout": _alignment(_split(supervised_holdout, holdout_lengths), holdout, target),
        },
        "kmeans_assignments": _split(kmeans, train_lengths),
    }


def _holdout_alignment(
    holdout: Sequence[alg.EventSource], relation: Sequence[torch.Tensor], message: Sequence[torch.Tensor],
    scales: alg.TrainScales, target: str,
) -> Mapping[str, Any]:
    joint, _ = alg._event_costs(holdout, relation, message, scales, qk_only=False)
    qk, _ = alg._event_costs(holdout, relation, message, scales, qk_only=True)
    return {"joint_R_plus_W": _alignment(joint, holdout, target), "qk_only_R": _alignment(qk, holdout, target)}


def _fit_from_assignments(
    train: Sequence[alg.EventSource], holdout: Sequence[alg.EventSource], target: str,
    assignments: Sequence[torch.Tensor], scales: alg.TrainScales, initialization: str,
    *, relation_rank: int = 32, message_rank: int = 64, max_outer_steps: int = 1,
) -> Mapping[str, Any]:
    assignments = tuple(labels.clone() for labels in assignments)
    relation = alg._spectral_relation_update(train, assignments, 2, relation_rank, scales)
    message, current = alg._exact_message_update(train, assignments, relation, None, 2, message_rank, scales)
    trajectory: list[Mapping[str, Any]] = [{
        "stage": "after_initial_M",
        "objective": current,
        "train": _alignment(assignments, train, target),
        "holdout": _holdout_alignment(holdout, relation, message, scales, target),
    }]
    converged = False
    failure = None
    for outer in range(max_outer_steps):
        start = current
        proposals = alg._spectral_relation_update(train, assignments, 2, relation_rank, scales)
        relation, spectral = alg._accept_spectral_relation(train, assignments, relation, message, proposals, scales)
        relation, relation_loss, refinement = alg._grassmann_refine_relation(train, assignments, relation, message, scales)
        message, message_loss = alg._exact_message_update(train, assignments, relation, message, 2, message_rank, scales)
        proposed, assignment_loss = alg._event_costs(train, relation, message, scales, qk_only=False)
        counts = [sum(int((labels == unit).sum()) for labels in proposed) for unit in range(2)]
        if min(counts) == 0:
            failure = f"E-step emptied a unit at outer {outer}"
            break
        moves = sum(int((old != new).sum()) for old, new in zip(assignments, proposed))
        relative = (start - assignment_loss) / max(1.0, abs(start))
        assignments = proposed
        current = assignment_loss
        trajectory.append({
            "stage": f"outer_{outer}",
            "objective": current,
            "post_spectral_objective": spectral,
            "post_relation_objective": relation_loss,
            "post_message_objective": message_loss,
            "relative_decrease": relative,
            "label_moves": moves,
            "event_counts": counts,
            "relation_refinement": refinement,
            "train": _alignment(assignments, train, target),
            "holdout": _holdout_alignment(holdout, relation, message, scales, target),
        })
        print(json.dumps({"stage": "oracle-event-fit", "outer": outer, "objective": current, "moves": moves, "train_ari": trajectory[-1]["train"]["ari"]}), flush=True)
        if moves == 0 and relative <= 1e-6:
            replay, replay_loss = alg._event_costs(train, relation, message, scales, qk_only=False)
            replay_moves = sum(int((old != new).sum()) for old, new in zip(assignments, replay))
            converged = replay_moves == 0
            current = replay_loss
            break
    return {
        "initialization": initialization,
        "converged": converged,
        "failure": failure,
        "iterations": len(trajectory) - 1,
        "final_objective": current,
        "trajectory": trajectory,
        "final_train": _alignment(assignments, train, target),
        "final_holdout": _holdout_alignment(holdout, relation, message, scales, target),
        "relation_projector_overlap": alg._overlap(relation),
        "message_projector_overlap": alg._overlap(message),
    }


def _prevalence(sources: Sequence[alg.EventSource], target: str) -> Mapping[str, Any]:
    truth = torch.cat([_truth(source, target) for source in sources])
    counts = torch.bincount(truth, minlength=2)
    energy = torch.zeros(2, dtype=DTYPE)
    for source in sources:
        event_energy = source.alpha[:, 1:].square() @ source.m[1:].square().sum(1)
        energy.scatter_add_(0, _truth(source, target), event_energy)
    return {
        "event_counts": counts.tolist(),
        "event_fraction": (counts.to(DTYPE) / counts.sum()).tolist(),
        "exact_pair_write_energy_fraction": (energy / energy.sum()).tolist(),
        "majority_accuracy": float(counts.max() / counts.sum()),
    }


def _run_head(raw: Sequence[SourceRows], target: str) -> Mapping[str, Any]:
    _v0, _complement, _c0 = _fit_common_span(raw)
    prepared = alg._prepare_sources(raw)
    train = [source for source in prepared if source.base.split == "train"]
    holdout = [source for source in prepared if source.base.split == "holdout"]
    scales = alg._fit_scales(train)
    geometry = _geometry(train, holdout, target)
    kmeans_assignments = geometry.pop("kmeans_assignments")
    result: dict[str, Any] = {
        "train_prevalence": _prevalence(train, target),
        "holdout_prevalence": _prevalence(holdout, target),
        "initialization_geometry": geometry,
        "kmeans_initial_train": _alignment(kmeans_assignments, train, target),
    }
    if target == "L1H1":
        oracle = tuple(_truth(source, target).clone() for source in train)
        result["oracle_fit"] = _fit_from_assignments(train, holdout, target, oracle, scales, "TRAIN semantic word/newline labels", max_outer_steps=5)
    return result


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    repo = Path(args.repo_root).resolve()
    _configure_imports(repo)
    from unifying_attention.experiments.unlearned_projector_real import load_registered_model_and_tokenizer, resolve_registered_snapshot
    from unifying_attention.unlearned_projector_gate import require_valid_smoke_report

    smoke = Path(args.smoke_report).resolve()
    require_valid_smoke_report(smoke)
    corpus_path = Path(args.corpus).resolve()
    corpus = json.loads(corpus_path.read_text())
    artifacts = resolve_registered_snapshot(allow_download=False)
    if corpus["model"]["revision"] != artifacts.revision:
        raise RuntimeError("Corpus/model revision mismatch")
    model, tokenizer = load_registered_model_and_tokenizer(artifacts, allow_download=False)
    try:
        l1, l1_audit = _extract_head(model, tokenizer, corpus, target="L1H1", layer=1, head=1, chunk_size=args.chunk_size, batch_size=args.batch_size)
        l5, l5_audit = _extract_head(model, tokenizer, corpus, target="L5H2", layer=5, head=2, chunk_size=args.chunk_size, batch_size=args.batch_size)
    finally:
        del model
        del tokenizer
    return {
        "schema_version": 1,
        "analysis": "investigate_label_algorithm_2",
        "development_only": True,
        "historical_test_accessed": False,
        "locked_test_accessed": False,
        "configuration": {"A": 2, "r": 32, "c": 64, "seed": 0, "objective": "mean_g[R_g,z_g + W_joint_g(z_g)]", "assignment": "one label per complete query event"},
        "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        "smoke_report": str(smoke),
        "extraction_audit": {"L1H1": l1_audit, "L5H2": l5_audit},
        "L1H1": _run_head(l1, "L1H1"),
        "L5H2": _run_head(l5, "L5H2"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--smoke-report", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    result = run(args)
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
