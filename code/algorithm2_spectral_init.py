"""Algorithm 2 with the spectral partition used only for TRAIN initialization.

This development-only ablation keeps Algorithm 2's complete R+W_joint M/E
loop and HOLD assignment unchanged.  The only intervention is the initial
TRAIN event labels: the product, attention-centred spectral kernel replaces
the six-coordinate PCA/K-means initializer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch


HERE = Path(__file__).resolve().parent
DEFAULT_EXPLORATORY = HERE.parent


def _configure_paths(exploratory_root: Path) -> None:
    for path in (
        exploratory_root / "code",
        exploratory_root / "spectral" / "code",
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


_configure_paths(DEFAULT_EXPLORATORY)
import algorithm2 as alg  # noqa: E402
import spectral_cluster as spectral  # noqa: E402


DTYPE = torch.float64


def _event_rows(corpus: Mapping[str, Any], target: str) -> Mapping[tuple[str, int], Mapping[str, Any]]:
    return {
        (str(row["source_id"]), int(row["query_position"])): row
        for row in corpus[target]["events"]
    }


def _build_train_bank(
    sources: Sequence[Any],
    *,
    target: str,
    event_rows: Mapping[tuple[str, int], Mapping[str, Any]],
    top_sources: int,
    max_events_per_source: int,
) -> spectral.EventBank:
    return spectral._build_bank(
        sources,
        target=target,
        split="train",
        top_sources=top_sources,
        max_events_per_source=max_events_per_source,
        event_rows=event_rows,
    )


def _kernel(
    left: spectral.EventBank,
    right: spectral.EventBank,
    *,
    block: int,
    symmetric: bool,
) -> torch.Tensor:
    tensor = spectral._tensor_gram(
        left, right, center="attention", block=block, symmetric=symmetric
    )
    tensor_left = spectral._tensor_self(left, center="attention")
    tensor_right = tensor_left if left is right else spectral._tensor_self(
        right, center="attention"
    )
    kernel, _left, _right = spectral._normalised_kernel(
        left.q @ right.q.T,
        tensor,
        left.q.square().sum(1),
        right.q.square().sum(1),
        tensor_left,
        tensor_right,
        mode="product",
        gamma=1.0,
    )
    if symmetric:
        kernel = 0.5 * (kernel + kernel.T)
        kernel.fill_diagonal_(1)
    return kernel


def _spectral_anchor_model(
    kernel: torch.Tensor,
    *,
    components: int,
    restarts: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Mapping[str, Any]]:
    affinity = kernel.clone()
    affinity.fill_diagonal_(0)
    degree = affinity.sum(1).clamp_min(1e-20)
    normalised = affinity / degree.sqrt()[:, None] / degree.sqrt()[None, :]
    eigenvalues, eigenvectors = torch.linalg.eigh(normalised)
    order = torch.argsort(eigenvalues, descending=True)
    rho = eigenvalues.index_select(0, order)[: components + 1]
    vectors = eigenvectors.index_select(1, order)[:, :components]
    embedding = vectors / vectors.norm(dim=1, keepdim=True).clamp_min(1e-30)
    labels, centers, inertia = spectral._kmeans(
        embedding, components, restarts=restarts, seed=seed + components
    )
    return labels, vectors, centers, {
        "top_normalised_affinity_eigenvalues": rho.tolist(),
        "A2_eigengap": float(rho[1] - rho[2]) if components == 2 else None,
        "anchor_inertia": inertia,
        "anchor_occupancy": torch.bincount(labels, minlength=components).tolist(),
    }


def _extend_to_full_train(
    anchors: spectral.EventBank,
    full: spectral.EventBank,
    anchor_kernel: torch.Tensor,
    anchor_labels: torch.Tensor,
    eigenvectors: torch.Tensor,
    centers: torch.Tensor,
    *,
    components: int,
    block: int,
) -> tuple[torch.Tensor, Mapping[str, Any]]:
    """Keep anchor labels and Nyström-extend only the remaining TRAIN events."""
    anchor_index = {
        (source_id, int(position)): index
        for index, (source_id, position) in enumerate(
            zip(anchors.source_ids, anchors.query_positions.tolist())
        )
    }
    full_keys = [
        (source_id, int(position))
        for source_id, position in zip(full.source_ids, full.query_positions.tolist())
    ]
    remaining_indices = torch.tensor(
        [i for i, key in enumerate(full_keys) if key not in anchor_index],
        dtype=torch.long,
    )
    labels = torch.full((full.count,), -1, dtype=torch.long)
    for full_index, key in enumerate(full_keys):
        if key in anchor_index:
            labels[full_index] = anchor_labels[anchor_index[key]]

    margin = torch.empty(0, dtype=DTYPE)
    if len(remaining_indices):
        remaining = full.subset(remaining_indices)
        cross = _kernel(remaining, anchors, block=block, symmetric=False)
        degree_train = anchor_kernel.clone()
        degree_train.fill_diagonal_(0)
        degree_train = degree_train.sum(1).clamp_min(1e-20)
        # Recover the eigenvalues paired with the retained TRAIN eigenvectors.
        affinity = anchor_kernel.clone()
        affinity.fill_diagonal_(0)
        affinity = affinity / degree_train.sqrt()[:, None] / degree_train.sqrt()[None, :]
        rho = torch.sum(eigenvectors * (affinity @ eigenvectors), dim=0)
        degree_new = cross.sum(1).clamp_min(1e-20)
        cross_affinity = cross / degree_new.sqrt()[:, None] / degree_train.sqrt()[None, :]
        embedding = cross_affinity @ eigenvectors[:, :components]
        embedding = embedding / rho[:components].clamp_min(1e-12)[None, :]
        embedding = embedding / embedding.norm(dim=1, keepdim=True).clamp_min(1e-30)
        distances = torch.cdist(embedding, centers).square()
        labels.index_copy_(0, remaining_indices, distances.argmin(1))
        if components > 1:
            ordered = distances.sort(1).values
            margin = ordered[:, 1] - ordered[:, 0]
    if bool((labels < 0).any()):
        raise RuntimeError("Not every TRAIN event received a spectral initial label.")
    return labels, {
        "anchor_events": anchors.count,
        "extended_train_events": int(len(remaining_indices)),
        "full_train_events": full.count,
        "extension_margin_median": float(margin.median()) if len(margin) else None,
        "extension_margin_p10": float(torch.quantile(margin, 0.10)) if len(margin) else None,
        "full_initial_occupancy": torch.bincount(labels, minlength=components).tolist(),
    }


def _split_bank_labels(
    bank: spectral.EventBank,
    labels: torch.Tensor,
    sources: Sequence[alg.EventSource],
) -> tuple[torch.Tensor, ...]:
    by_event = {
        (source_id, int(position)): int(label)
        for source_id, position, label in zip(
            bank.source_ids, bank.query_positions.tolist(), labels.tolist()
        )
    }
    if len(by_event) != bank.count:
        raise RuntimeError("TRAIN event identifiers are not unique.")
    result = []
    for source in sources:
        local = torch.tensor(
            [
                by_event[(str(source.base.source_id), int(position))]
                for position in source.base.query_positions.tolist()
            ],
            dtype=torch.long,
        )
        result.append(local)
    if sum(len(row) for row in result) != bank.count:
        raise RuntimeError("Spectral labels do not cover Algorithm 2's TRAIN bank.")
    return tuple(result)


def _alignment(
    sources: Sequence[alg.EventSource], labels: Sequence[torch.Tensor]
) -> Mapping[str, Any]:
    discovered = torch.cat(tuple(labels))
    semantic = torch.cat([source.base.semantic_labels for source in sources])
    subtype = torch.cat([source.base.subtype_labels for source in sources])
    components = int(discovered.max()) + 1
    return {
        "event_occupancy_fraction_by_unit": (
            torch.bincount(discovered, minlength=components).to(DTYPE) / len(discovered)
        ).tolist(),
        "semantic_event_ari": alg._ari(discovered, semantic),
        "semantic_event_permutation_accuracy": alg._permutation_accuracy(
            discovered, semantic
        ),
        "subtype_event_ari": alg._ari(discovered, subtype),
        "subtype_event_permutation_accuracy": alg._permutation_accuracy(
            discovered, subtype
        ),
    }


def _fit_from_assignments(
    sources: Sequence[alg.EventSource],
    initial_assignments: Sequence[torch.Tensor],
    *,
    components: int,
    relation_rank: int,
    message_rank: int,
    scales: alg.TrainScales,
) -> alg.FitState:
    assignments = tuple(labels.clone() for labels in initial_assignments)
    counts = [
        sum(int((labels == unit).sum()) for labels in assignments)
        for unit in range(components)
    ]
    if min(counts) == 0:
        raise RuntimeError("Spectral initialization produced an empty TRAIN unit.")
    relation = alg._spectral_relation_update(
        sources, assignments, components, relation_rank, scales
    )
    message, current_loss = alg._exact_message_update(
        sources, assignments, relation, None, components, message_rank, scales
    )
    history: list[Mapping[str, Any]] = []
    converged = False
    for outer in range(alg.OUTER_MAX_STEPS):
        outer_start = current_loss
        proposals = alg._spectral_relation_update(
            sources, assignments, components, relation_rank, scales
        )
        relation, spectral_loss = alg._accept_spectral_relation(
            sources, assignments, relation, message, proposals, scales
        )
        relation, relation_loss, refinement = alg._grassmann_refine_relation(
            sources, assignments, relation, message, scales
        )
        if relation_loss > outer_start + alg.MONOTONE_RTOL * max(1.0, abs(outer_start)):
            raise RuntimeError("Relation M-step increased the objective.")
        message, message_loss = alg._exact_message_update(
            sources,
            assignments,
            relation,
            message,
            components,
            message_rank,
            scales,
        )
        if message_loss > relation_loss + alg.MONOTONE_RTOL * max(1.0, abs(relation_loss)):
            raise RuntimeError("Message M-step increased the objective.")
        proposed, assignment_loss = alg._event_costs(
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
        if assignment_loss > message_loss + alg.MONOTONE_RTOL * max(1.0, abs(message_loss)):
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
        print(
            json.dumps({"stage": "event-fit-spectral-init", "A": components, **record}),
            flush=True,
        )
        assignments = proposed
        current_loss = assignment_loss
        if moves == 0 and relative_decrease <= 1e-6:
            converged = True
            break
    if not converged:
        raise RuntimeError(f"A={components} hit the outer-iteration cap.")
    replay, replay_loss = alg._event_costs(
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
    return alg.FitState(
        tuple(relation), tuple(message), replay, tuple(history), True
    )


def _run_head(
    raw_sources: list[Any],
    *,
    target: str,
    corpus: Mapping[str, Any],
    relation_rank: int,
    message_rank: int,
    top_sources: int,
    max_anchor_events_per_source: int,
    kernel_block: int,
    kmeans_restarts: int,
    spectral_seed: int,
) -> Mapping[str, Any]:
    v0, complement, c0 = alg._fit_common_span(raw_sources)
    if message_rank > complement.shape[1]:
        raise RuntimeError("Requested message rank exceeds the V0 complement.")
    prepared = alg._prepare_sources(raw_sources)
    train = [source for source in prepared if source.base.split == "train"]
    holdout = [source for source in prepared if source.base.split == "holdout"]
    scales = alg._fit_scales(train)

    rows = _event_rows(corpus, target)
    anchors = _build_train_bank(
        raw_sources,
        target=target,
        event_rows=rows,
        top_sources=top_sources,
        max_events_per_source=max_anchor_events_per_source,
    )
    full = _build_train_bank(
        raw_sources,
        target=target,
        event_rows=rows,
        top_sources=top_sources,
        max_events_per_source=10**9,
    )
    expected = sum(source.event_count for source in train)
    if full.count != expected:
        raise RuntimeError("Full spectral TRAIN bank differs from Algorithm 2 TRAIN events.")
    anchor_kernel = _kernel(anchors, anchors, block=kernel_block, symmetric=True)
    anchor_labels, vectors, centers, spectral_record = _spectral_anchor_model(
        anchor_kernel,
        components=2,
        restarts=kmeans_restarts,
        seed=spectral_seed,
    )
    full_labels, extension = _extend_to_full_train(
        anchors,
        full,
        anchor_kernel,
        anchor_labels,
        vectors,
        centers,
        components=2,
        block=kernel_block,
    )
    initial_a2 = _split_bank_labels(full, full_labels, train)
    initial_by_a = {
        1: tuple(torch.zeros(source.event_count, dtype=torch.long) for source in train),
        2: initial_a2,
    }

    fits = []
    for components in (1, 2):
        initial = initial_by_a[components]
        state = _fit_from_assignments(
            train,
            initial,
            components=components,
            relation_rank=relation_rank,
            message_rank=message_rank,
            scales=scales,
        )
        train_metrics = alg._metric_record(
            train,
            state.assignments,
            state.relation,
            state.message,
            scales,
            v0,
            complement,
            assignment_mode="joint",
        )
        hold_joint_labels, _ = alg._event_costs(
            holdout, state.relation, state.message, scales, qk_only=False
        )
        hold_qk_labels, _ = alg._event_costs(
            holdout, state.relation, state.message, scales, qk_only=True
        )
        hold_joint = alg._metric_record(
            holdout,
            hold_joint_labels,
            state.relation,
            state.message,
            scales,
            v0,
            complement,
            assignment_mode="joint",
        )
        hold_qk = alg._metric_record(
            holdout,
            hold_qk_labels,
            state.relation,
            state.message,
            scales,
            v0,
            complement,
            assignment_mode="qk_only",
        )
        initial_flat = torch.cat(initial)
        final_flat = torch.cat(state.assignments)
        fits.append(
            {
                "A": components,
                "r": relation_rank,
                "c": message_rank,
                "converged": state.converged,
                "iterations": len(
                    [row for row in state.history if isinstance(row.get("outer"), int)]
                ),
                "history": state.history,
                "initial_train_alignment": _alignment(train, initial),
                "final_vs_initial_ari": alg._ari(final_flat, initial_flat),
                "final_vs_initial_permutation_accuracy": alg._permutation_accuracy(
                    final_flat, initial_flat
                ),
                "relation_projector_overlap": alg._overlap(state.relation),
                "message_projector_overlap": alg._overlap(state.message),
                "train_joint": train_metrics,
                "holdout_joint": hold_joint,
                "holdout_qk_only": hold_qk,
                "routing_regret_W_joint_pooled": (
                    hold_qk["W_joint_pooled"] - hold_joint["W_joint_pooled"]
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
        "spectral_initialization": {
            "kernel": "product_attention_center",
            "spectral_anchor": spectral_record,
            "train_extension": extension,
            "holdout_used": False,
        },
        "fits": fits,
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    exploratory = Path(args.exploratory_root).resolve()
    _configure_paths(exploratory)
    repo_root = Path(args.repo_root).resolve()
    alg._configure_imports(repo_root)
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
        l1, l1_audit = alg._extract_head(
            model,
            tokenizer,
            corpus,
            target="L1H1",
            layer=1,
            head=1,
            chunk_size=args.chunk_size,
            batch_size=args.batch_size,
        )
        l5, l5_audit = alg._extract_head(
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

    common = {
        "corpus": corpus,
        "relation_rank": args.relation_rank,
        "message_rank": args.message_rank,
        "top_sources": args.top_sources,
        "max_anchor_events_per_source": args.max_anchor_events_per_source,
        "kernel_block": args.kernel_block,
        "kmeans_restarts": args.kmeans_restarts,
        "spectral_seed": args.spectral_seed,
    }
    return {
        "schema_version": 1,
        "analysis": "algorithm2_spectral_train_initialization_ablation",
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
            "spectral_seed": args.spectral_seed,
            "spectral_kmeans_restarts": args.kmeans_restarts,
            "spectral_kernel": "product_attention_center",
            "spectral_anchor_events_per_source": args.max_anchor_events_per_source,
            "spectral_top_sources": args.top_sources,
            "fit_objective": "mean_g [R_g,z + W_joint_g(z)]",
            "train_updates": "unchanged Algorithm 2 alternating M_U, exact M_V, complete R+W E-step",
            "holdout_assignment": "unchanged Algorithm 2 argmin_a R_g,a + W_joint_g(a); QK-only diagnostic also reported",
            "holdout_in_spectral_initialization": False,
        },
        "metric_definitions": {
            "R_macro": "mean event Q/K subspace residual R_g,z",
            "P_pooled": "pooled exact-attention source-message projection error",
            "W_joint_pooled": "pooled executed innovation-write error with reconstructed attention and routed projected messages",
            "W_msg_pooled": "pooled message compression error at exact attention",
            "W_rel_pooled": "pooled relation-side write error with exact innovation messages",
            "full_write_nmse_pooled": "complete BOS+common+innovation write squared error divided by complete write energy",
            "full_write_cosine_pooled": "cosine of stacked complete exact and predicted writes",
            "attention_kl_mean": "mean complete-row KL(alpha_exact || alpha_reconstructed)",
            "semantic_event_ari_and_accuracy": "agreement with post-hoc behavior labels; labels never enter fitting",
            "final_vs_initial_ari": "permutation-invariant retention of spectral TRAIN initialization after Algorithm 2 EM",
            "joint_assignment": "argmin_a R_g,a + W_joint_g(a) with frozen TRAIN parameters",
            "qk_only_assignment": "argmin_a R_g,a; diagnostic only",
        },
        "dependencies": {
            "algorithm2_sha256": hashlib.sha256(
                (exploratory / "code" / "algorithm2.py").read_bytes()
            ).hexdigest(),
            "spectral_cluster_sha256": hashlib.sha256(
                (exploratory / "spectral" / "code" / "spectral_cluster.py").read_bytes()
            ).hexdigest(),
        },
        "extraction_audit": {"L1H1": l1_audit, "L5H2": l5_audit},
        "L1H1": _run_head(l1, target="L1H1", **common),
        "L5H2": _run_head(l5, target="L5H2", **common),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--exploratory-root", type=Path, default=DEFAULT_EXPLORATORY)
    parser.add_argument("--smoke-report", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--relation-rank", type=int, default=32)
    parser.add_argument("--message-rank", type=int, default=64)
    parser.add_argument("--top-sources", type=int, default=8)
    parser.add_argument("--max-anchor-events-per-source", type=int, default=6)
    parser.add_argument("--kernel-block", type=int, default=48)
    parser.add_argument("--kmeans-restarts", type=int, default=20)
    parser.add_argument("--spectral-seed", type=int, default=1729)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    numeric = (
        args.chunk_size,
        args.batch_size,
        args.threads,
        args.relation_rank,
        args.message_rank,
        args.top_sources,
        args.max_anchor_events_per_source,
        args.kernel_block,
        args.kmeans_restarts,
    )
    if min(numeric) <= 0:
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
