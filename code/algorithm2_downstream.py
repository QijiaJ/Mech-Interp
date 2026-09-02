"""Downstream causal interpretation of the spectral-initialized Algorithm 2 fits.

This development-only diagnostic re-fits exactly the two biological hypotheses
already studied on the frozen natural corpus: L1H1 with A=2 and L5H2 with A=1,
both at r=32,c=64.  No locked or historical split is read.  It then transports
exact and fitted HOLD writes through the unchanged later Gemma layers.

The primary intervention isolates the innovation message governed by V_a.  At
one annotated query position three identical prompt copies receive: the exact
model state, exact-innovation ablation, and fitted-innovation substitution.
The secondary intervention repeats this with the complete head write.  Both
exact and fitted effects are therefore measured relative to the same ablated
state.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch


DTYPE = torch.float64
DEFAULT_EXPLORATORY = Path(__file__).resolve().parents[1]


def _configure(exploratory: Path, repo_root: Path) -> None:
    for path in (
        exploratory / "code",
        exploratory / "spectral" / "code",
        repo_root,
        repo_root / "unifying_algorithm",
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _normalised_squared_error(predicted: torch.Tensor, exact: torch.Tensor) -> float:
    return float((predicted - exact).square().sum() / exact.square().sum().clamp_min(1e-30))


def _cosine(predicted: torch.Tensor, exact: torch.Tensor) -> float:
    return float(
        (predicted * exact).sum()
        / (predicted.square().sum() * exact.square().sum()).sqrt().clamp_min(1e-30)
    )


def _r90(rows: torch.Tensor) -> int:
    if not len(rows) or float(rows.square().sum()) == 0.0:
        return 0
    singular = torch.linalg.svdvals(rows.to(DTYPE))
    energy = singular.square()
    return int((energy.cumsum(0) < 0.90 * energy.sum()).sum()) + 1


def _decoded_tokens(
    values: torch.Tensor,
    tokenizer: Any,
    *,
    largest: bool,
    count: int,
) -> list[Mapping[str, Any]]:
    special = set(getattr(tokenizer, "all_special_ids", ()))
    result = []
    for token_id in values.argsort(descending=largest).tolist():
        if int(token_id) in special:
            continue
        token = tokenizer.decode([int(token_id)]).replace("\n", "\\n")
        if not token.replace("\\n", "").strip():
            continue
        result.append({"token": token, "change": float(values[token_id])})
        if len(result) == count:
            break
    return result


def _top_sources(source: Any, event: int, *, count: int = 3) -> list[Mapping[str, Any]]:
    alpha = source.alpha[event]
    valid = source.mask[event].clone()
    valid[0] = False
    positions = valid.nonzero(as_tuple=True)[0]
    order = torch.argsort(alpha.index_select(0, positions), descending=True, stable=True)
    result = []
    for local in order[:count].tolist():
        position = int(positions[local])
        result.append(
            {
                "position": position,
                "token": source.base.tokens[position].replace("\n", "\\n"),
                "attention": float(alpha[position]),
            }
        )
    return result


def _top_predicted_sources(
    source: Any,
    event: int,
    relation: torch.Tensor,
    alg: Any,
    *,
    count: int = 3,
) -> list[Mapping[str, Any]]:
    alpha = alg._predicted_attention(source, relation)[event]
    valid = source.mask[event].clone()
    valid[0] = False
    positions = valid.nonzero(as_tuple=True)[0]
    order = torch.argsort(alpha.index_select(0, positions), descending=True, stable=True)
    result = []
    for local in order[:count].tolist():
        position = int(positions[local])
        result.append(
            {
                "position": position,
                "token": source.base.tokens[position].replace("\n", "\\n"),
                "attention": float(alpha[position]),
            }
        )
    return result


def _event_writes(
    source: Any,
    event: int,
    unit: int,
    *,
    relation: Sequence[torch.Tensor],
    message: Sequence[torch.Tensor],
    v0: torch.Tensor,
    complement: torch.Tensor,
    basis: torch.Tensor,
    alg: Any,
) -> Mapping[str, torch.Tensor]:
    """Return exact/fitted innovation and complete writes in residual space."""
    exact_alpha = source.alpha[event].to(DTYPE)
    predicted_alpha = alg._predicted_attention(source, relation[unit])[event]
    original = source.base.messages.to(DTYPE)
    exact_complete_coordinate = exact_alpha @ original

    exact_innovation_coordinate = source.exact_y[event] @ complement.T
    predicted_complement = predicted_alpha[1:] @ source.m[1:]
    fitted_innovation_coordinate = (
        (predicted_complement @ message[unit]) @ message[unit].T
    ) @ complement.T

    common_coordinates = original @ v0
    fitted_common_coordinate = (
        predicted_alpha[1:] @ common_coordinates[1:]
    ) @ v0.T
    fitted_bos_coordinate = predicted_alpha[0] * original[0]
    fitted_complete_coordinate = (
        fitted_bos_coordinate + fitted_common_coordinate + fitted_innovation_coordinate
    )

    return {
        "exact_innovation": exact_innovation_coordinate @ basis.T,
        "fitted_innovation": fitted_innovation_coordinate @ basis.T,
        "exact_complete": exact_complete_coordinate @ basis.T,
        "fitted_complete": fitted_complete_coordinate @ basis.T,
    }


@torch.no_grad()
def _transport(
    model: Any,
    tokenizer: Any,
    *,
    input_ids: torch.Tensor,
    query_position: int,
    next_token_id: int,
    layer: int,
    exact_write: torch.Tensor,
    fitted_write: torch.Tensor,
) -> Mapping[str, Any]:
    """Compare exact and fitted write additions relative to one ablated state."""
    ids = input_ids[None].expand(3, -1).clone()
    module = model.model.layers[layer].self_attn

    def edit_attention(_module: Any, _inputs: Any, output: Any) -> Any:
        values = output[0] if isinstance(output, tuple) else output
        edited = values.clone()
        exact = exact_write.to(device=edited.device, dtype=edited.dtype)
        fitted = fitted_write.to(device=edited.device, dtype=edited.dtype)
        # row 0: exact model; row 1: exact write removed; row 2: fitted replaces exact.
        edited[1, query_position] -= exact
        edited[2, query_position] += fitted - exact
        if isinstance(output, tuple):
            return (edited,) + tuple(output[1:])
        return edited

    handle = module.register_forward_hook(edit_attention)
    try:
        logits = model(input_ids=ids, use_cache=False).logits.detach().to(torch.float64)
    finally:
        handle.remove()
    plain_logits = model(input_ids=ids[:1], use_cache=False).logits.detach().to(torch.float64)
    row0_hook_error = float((plain_logits[0] - logits[0]).abs().max())
    if query_position > 0:
        prefix_error = max(
            float((logits[row, :query_position] - logits[0, :query_position]).abs().max())
            for row in (1, 2)
        )
    else:
        prefix_error = 0.0
    readout = query_position
    exact_logits, ablated_logits, fitted_logits = logits[:, readout]
    exact_logprob = exact_logits.log_softmax(-1)
    ablated_logprob = ablated_logits.log_softmax(-1)
    fitted_logprob = fitted_logits.log_softmax(-1)
    exact_probability = exact_logprob.exp()
    ablated_probability = ablated_logprob.exp()
    fitted_probability = fitted_logprob.exp()
    exact_probability_effect = exact_probability - ablated_probability
    fitted_probability_effect = fitted_probability - ablated_probability
    exact_logprob_effect = exact_logprob - ablated_logprob
    fitted_logprob_effect = fitted_logprob - ablated_logprob
    exact_signature = exact_probability.sqrt() * exact_logprob_effect
    fitted_signature = exact_probability.sqrt() * fitted_logprob_effect
    exact_top = set(torch.topk(exact_probability_effect.abs(), 20).indices.tolist())
    fitted_top = set(torch.topk(fitted_probability_effect.abs(), 20).indices.tolist())
    return {
        "probability_effect_cosine": _cosine(fitted_probability_effect, exact_probability_effect),
        "probability_effect_nmse": _normalised_squared_error(fitted_probability_effect, exact_probability_effect),
        "logprob_effect_cosine": _cosine(fitted_logprob_effect, exact_logprob_effect),
        "transported_signature_cosine": _cosine(fitted_signature, exact_signature),
        "transported_signature_nmse": _normalised_squared_error(fitted_signature, exact_signature),
        "exact_signature_norm": float(exact_signature.norm()),
        "fitted_signature_norm": float(fitted_signature.norm()),
        "top20_absolute_effect_jaccard": len(exact_top & fitted_top) / len(exact_top | fitted_top),
        "cached_next_token_id": int(next_token_id),
        "cached_next_token_base_probability": float(exact_probability[next_token_id]),
        "cached_next_token_ablated_probability": float(ablated_probability[next_token_id]),
        "cached_next_token_fitted_substitution_probability": float(fitted_probability[next_token_id]),
        "cached_next_token_exact_probability_effect": float(
            exact_probability_effect[next_token_id]
        ),
        "cached_next_token_fitted_probability_effect": float(
            fitted_probability_effect[next_token_id]
        ),
        "row0_hook_vs_unhooked_logits_max_abs_error": row0_hook_error,
        "causal_prefix_logits_max_abs_error": prefix_error,
        "exact_promoted_tokens": _decoded_tokens(
            exact_probability_effect, tokenizer, largest=True, count=6
        ),
        "exact_suppressed_tokens": _decoded_tokens(
            exact_probability_effect, tokenizer, largest=False, count=6
        ),
        "fitted_promoted_tokens": _decoded_tokens(
            fitted_probability_effect, tokenizer, largest=True, count=6
        ),
        "fitted_suppressed_tokens": _decoded_tokens(
            fitted_probability_effect, tokenizer, largest=False, count=6
        ),
    }


def _choose_examples(
    holdout: Sequence[Any],
    labels: Sequence[torch.Tensor],
    *,
    target: str,
) -> list[tuple[int, int, int]]:
    """Choose high-energy, source-diverse HOLD events for each natural family."""
    by_family: dict[str, list[tuple[float, int, int, int, str]]] = defaultdict(list)
    for source_index, (source, local_labels) in enumerate(zip(holdout, labels)):
        for event, (family, unit) in enumerate(zip(source.base.families, local_labels.tolist())):
            energy = float(source.exact_y[event].square().sum())
            by_family[family].append(
                (energy, source_index, event, int(unit), str(source.base.source_id))
            )
    result: list[tuple[int, int, int]] = []
    if target == "L1H1":
        desired = {
            "alphabetic_word": 3,
            "four_digit_year": 3,
            "other_number": 3,
            "identifier": 3,
            "newline_boundary": 3,
        }
    else:
        desired = {"strict_induction": 3, "tokenization_tolerant_induction": 3}
    for family, count in desired.items():
        seen_sources: set[str] = set()
        for _energy, source_index, event, unit, source_id in sorted(
            by_family[family], reverse=True
        ):
            if source_id in seen_sources:
                continue
            result.append((source_index, event, unit))
            seen_sources.add(source_id)
            if len(seen_sources) == count:
                break
    return result


def _usage_ranks(
    sources: Sequence[Any],
    labels: Sequence[torch.Tensor],
    relation: Sequence[torch.Tensor],
    message: Sequence[torch.Tensor],
    alg: Any,
) -> list[Mapping[str, Any]]:
    records = []
    for unit, (u, v) in enumerate(zip(relation, message)):
        q_coordinates = []
        k_coordinates = []
        exact_writes = []
        fitted_writes = []
        projected_source_messages = []
        events = 0
        for source, local_labels in zip(sources, labels):
            indices = (local_labels == unit).nonzero(as_tuple=True)[0]
            if not len(indices):
                continue
            events += len(indices)
            q_coordinates.append(source.q.index_select(0, indices) @ u)
            alpha = source.alpha.index_select(0, indices)[:, 1:]
            # One attention-weighted key coordinate per event is the relation
            # activity actually presented to the event-level router.
            k_coordinates.append(
                (alpha @ (source.k[1:] @ u))
                / alpha.sum(1, keepdim=True).clamp_min(1e-30)
            )
            exact_writes.append(source.exact_y.index_select(0, indices))
            # This is the source-message population seen by Algorithm 2's
            # alpha^2-weighted P diagnostic, represented inside V_a.
            source_weight = alpha.square().sum(0).sqrt()
            projected_source_messages.append(
                source_weight[:, None] * (source.m[1:] @ v)
            )
            alpha_hat = alg._predicted_attention(source, u).index_select(0, indices)
            b = alpha_hat[:, 1:] @ source.m[1:]
            fitted_writes.append((b @ v) @ v.T)
        q_rows = torch.cat(q_coordinates) if q_coordinates else torch.empty(0, u.shape[1])
        k_rows = torch.cat(k_coordinates) if k_coordinates else torch.empty(0, u.shape[1])
        exact_rows = torch.cat(exact_writes) if exact_writes else torch.empty(0, v.shape[0])
        fitted_rows = torch.cat(fitted_writes) if fitted_writes else torch.empty(0, v.shape[0])
        source_message_rows = (
            torch.cat(projected_source_messages)
            if projected_source_messages
            else torch.empty(0, v.shape[1])
        )
        records.append(
            {
                "unit": unit,
                "events": events,
                "nominal_relation_rank": int(u.shape[1]),
                "nominal_message_rank": int(v.shape[1]),
                "q_coordinate_r90": _r90(q_rows),
                "attended_key_coordinate_r90": _r90(k_rows),
                "projected_source_message_coordinate_r90": _r90(source_message_rows),
                "exact_innovation_write_r90": _r90(exact_rows),
                "fitted_innovation_write_r90": _r90(fitted_rows),
            }
        )
    return records


def _aggregate_transport(examples: Sequence[Mapping[str, Any]], key: str) -> Mapping[str, Any]:
    rows = [example[key] for example in examples]
    mean_fields = (
        "probability_effect_cosine",
        "probability_effect_nmse",
        "logprob_effect_cosine",
        "transported_signature_cosine",
        "transported_signature_nmse",
        "exact_signature_norm",
        "fitted_signature_norm",
        "top20_absolute_effect_jaccard",
    )
    result = {
        field: sum(float(row[field]) for row in rows) / max(len(rows), 1)
        for field in mean_fields
    }
    for field in (
        "row0_hook_vs_unhooked_logits_max_abs_error",
        "causal_prefix_logits_max_abs_error",
    ):
        result[field] = max((float(row[field]) for row in rows), default=0.0)
    return result


def _run_target(
    model: Any,
    tokenizer: Any,
    corpus: Mapping[str, Any],
    raw_sources: Sequence[Any],
    *,
    target: str,
    layer: int,
    head: int,
    basis: torch.Tensor,
    alg: Any,
    spectral_init: Any,
    relation_rank: int,
    message_rank: int,
    top_sources: int,
    max_anchor_events_per_source: int,
    kernel_block: int,
    kmeans_restarts: int,
    spectral_seed: int,
    artifact_path: Path,
) -> Mapping[str, Any]:
    v0, complement, c0 = alg._fit_common_span(raw_sources)
    prepared = alg._prepare_sources(raw_sources)
    train = [source for source in prepared if source.base.split == "train"]
    holdout = [source for source in prepared if source.base.split == "holdout"]
    scales = alg._fit_scales(train)
    components = 2 if target == "L1H1" else 1

    if components == 2:
        rows = spectral_init._event_rows(corpus, target)
        anchors = spectral_init._build_train_bank(
            raw_sources,
            target=target,
            event_rows=rows,
            top_sources=top_sources,
            max_events_per_source=max_anchor_events_per_source,
        )
        full = spectral_init._build_train_bank(
            raw_sources,
            target=target,
            event_rows=rows,
            top_sources=top_sources,
            max_events_per_source=10**9,
        )
        anchor_kernel = spectral_init._kernel(
            anchors, anchors, block=kernel_block, symmetric=True
        )
        anchor_labels, vectors, centers, spectral_record = (
            spectral_init._spectral_anchor_model(
                anchor_kernel,
                components=components,
                restarts=kmeans_restarts,
                seed=spectral_seed,
            )
        )
        full_labels, extension = spectral_init._extend_to_full_train(
            anchors,
            full,
            anchor_kernel,
            anchor_labels,
            vectors,
            centers,
            components=components,
            block=kernel_block,
        )
        initial = spectral_init._split_bank_labels(full, full_labels, train)
        initialization = {"spectral": spectral_record, "extension": extension}
    else:
        initial = tuple(
            torch.zeros(source.event_count, dtype=torch.long) for source in train
        )
        initialization = {"spectral": "A=1 has the unique assignment"}

    state = spectral_init._fit_from_assignments(
        train,
        initial,
        components=components,
        relation_rank=relation_rank,
        message_rank=message_rank,
        scales=scales,
    )
    hold_labels, _ = alg._event_costs(
        holdout, state.relation, state.message, scales, qk_only=False
    )
    hold_metrics = alg._metric_record(
        holdout,
        hold_labels,
        state.relation,
        state.message,
        scales,
        v0,
        complement,
        assignment_mode="joint",
    )

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "target": target,
            "layer": layer,
            "head": head,
            "c0": c0,
            "v0": v0,
            "complement": complement,
            "relation": state.relation,
            "message": state.message,
            "train_assignments": state.assignments,
            "hold_assignments": hold_labels,
            "scales": scales,
        },
        artifact_path,
    )

    source_text = {str(row["source_id"]): str(row["text"]) for row in corpus["sources"]}
    selected = _choose_examples(holdout, hold_labels, target=target)
    examples = []
    for source_index, event, unit in selected:
        source = holdout[source_index]
        source_id = str(source.base.source_id)
        text = source_text[source_id]
        ids = torch.tensor(
            tokenizer(text, add_special_tokens=True)["input_ids"], dtype=torch.long
        )
        retokenized_tokens = tuple(
            tokenizer.decode([int(token_id)], skip_special_tokens=False)
            for token_id in ids.tolist()
        )
        if retokenized_tokens != source.base.tokens:
            raise RuntimeError(f"Retokenized prompt differs from cached source {source_id}.")
        query = int(source.base.query_positions[event])
        if query + 1 >= len(ids):
            raise RuntimeError(f"Active query lacks a cached next token in {source_id}.")
        next_token_id = int(ids[query + 1])
        writes = _event_writes(
            source,
            event,
            unit,
            relation=state.relation,
            message=state.message,
            v0=v0,
            complement=complement,
            basis=basis,
            alg=alg,
        )
        innovation = _transport(
            model,
            tokenizer,
            input_ids=ids,
            query_position=query,
            next_token_id=next_token_id,
            layer=layer,
            exact_write=writes["exact_innovation"],
            fitted_write=writes["fitted_innovation"],
        )
        complete = _transport(
            model,
            tokenizer,
            input_ids=ids,
            query_position=query,
            next_token_id=next_token_id,
            layer=layer,
            exact_write=writes["exact_complete"],
            fitted_write=writes["fitted_complete"],
        )
        examples.append(
            {
                "source_id": source_id,
                "domain": source.base.domain,
                "family": source.base.families[event],
                "unit": int(unit),
                "prompt": text,
                "retokenized_prompt_matches_cache": True,
                "query_position": query,
                "query_token": source.base.tokens[query].replace("\n", "\\n"),
                "next_token": (
                    source.base.tokens[query + 1].replace("\n", "\\n")
                    if query + 1 < len(source.base.tokens)
                    else None
                ),
                "exact_top_sources": _top_sources(source, event),
                "fitted_top_sources": _top_predicted_sources(
                    source, event, state.relation[unit], alg
                ),
                "exact_innovation_norm": float(writes["exact_innovation"].norm()),
                "fitted_innovation_norm": float(writes["fitted_innovation"].norm()),
                "innovation_write_cosine": _cosine(
                    writes["fitted_innovation"], writes["exact_innovation"]
                ),
                "complete_write_cosine": _cosine(
                    writes["fitted_complete"], writes["exact_complete"]
                ),
                "innovation_transport": innovation,
                "complete_transport": complete,
            }
        )

    # Unit numbering is arbitrary; record its biological composition explicitly.
    composition = []
    for unit in range(components):
        counts: dict[str, int] = defaultdict(int)
        for source, local_labels in zip(holdout, hold_labels):
            for family, label in zip(source.base.families, local_labels.tolist()):
                if int(label) == unit:
                    counts[family] += 1
        composition.append({"unit": unit, "family_counts": dict(sorted(counts.items()))})

    return {
        "target": target,
        "layer": layer,
        "head": head,
        "A": components,
        "nominal_r": relation_rank,
        "nominal_c": message_rank,
        "common_rank_c0": c0,
        "initialization": initialization,
        "fit_iterations": len(
            [row for row in state.history if isinstance(row.get("outer"), int)]
        ),
        "holdout_metrics": hold_metrics,
        "holdout_unit_composition": composition,
        "holdout_usage_ranks": _usage_ranks(
            holdout, hold_labels, state.relation, state.message, alg
        ),
        "transport_aggregate": {
            "innovation": _aggregate_transport(examples, "innovation_transport"),
            "complete": _aggregate_transport(examples, "complete_transport"),
        },
        "examples": examples,
        "fit_artifact": str(artifact_path),
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    exploratory = args.exploratory_root.resolve()
    repo_root = args.repo_root.resolve()
    _configure(exploratory, repo_root)
    import algorithm1
    import algorithm2 as alg
    import algorithm2_spectral_init as spectral_init
    from unifying_attention.experiments.unlearned_projector_real import (
        load_registered_model_and_tokenizer,
        resolve_registered_snapshot,
    )
    from unifying_attention.unlearned_projector_data import (
        compute_message_basis,
        head_output_block,
    )
    from unifying_attention.unlearned_projector_gate import require_valid_smoke_report

    require_valid_smoke_report(args.smoke_report.resolve())
    corpus_path = args.corpus.resolve()
    corpus = json.loads(corpus_path.read_text())
    artifacts = resolve_registered_snapshot(allow_download=False)
    if corpus["model"]["revision"] != artifacts.revision:
        raise RuntimeError("Corpus/model revision mismatch.")
    model, tokenizer = load_registered_model_and_tokenizer(
        artifacts, allow_download=False
    )
    model.to(device="cpu", dtype=torch.float32)
    model.eval()
    results = {}
    audits = {}
    try:
        for target, layer, head in (("L1H1", 1, 1), ("L5H2", 5, 2)):
            raw_sources, audit = algorithm1._extract_head(
                model,
                tokenizer,
                corpus,
                target=target,
                layer=layer,
                head=head,
                chunk_size=args.chunk_size,
                batch_size=args.batch_size,
            )
            audits[target] = audit
            basis = compute_message_basis(
                head_output_block(model, layer_idx=layer, head=head),
                expected_rank=256,
            ).basis
            results[target] = _run_target(
                model,
                tokenizer,
                corpus,
                raw_sources,
                target=target,
                layer=layer,
                head=head,
                basis=basis,
                alg=alg,
                spectral_init=spectral_init,
                relation_rank=args.relation_rank,
                message_rank=args.message_rank,
                top_sources=args.top_sources,
                max_anchor_events_per_source=args.max_anchor_events_per_source,
                kernel_block=args.kernel_block,
                kmeans_restarts=args.kmeans_restarts,
                spectral_seed=args.spectral_seed,
                artifact_path=args.artifact_dir.resolve() / f"{target}_fit.pt",
            )
    finally:
        del model
        del tokenizer

    return {
        "schema_version": 1,
        "analysis": "algorithm2_spectral_downstream_causal_interpretation",
        "development_only": True,
        "locked_test_accessed": False,
        "historical_test_accessed": False,
        "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        "configuration": {
            "L1H1": {"A": 2, "r": args.relation_rank, "c": args.message_rank},
            "L5H2": {"A": 1, "r": args.relation_rank, "c": args.message_rank},
            "initializer": "product attention-centered spectral TRAIN labels for L1H1; unique A=1 labels for L5H2",
            "fitter": "unchanged Algorithm 2 R+W_joint generalized hard EM",
            "holdout_assignment": "argmin_a R_g,a+W_joint_g(a)",
            "intervention": "exact state, exact-write ablation, fitted-for-exact substitution at the selected head's raw self-attention output",
            "readout": "final next-token distribution at the same annotated query position",
        },
        "rank_definitions": {
            "nominal": "stored orthonormal frame width",
            "q_coordinate_r90": "90% pooled singular-value energy rank of q_g^T U_a on assigned HOLD events",
            "attended_key_coordinate_r90": "90% pooled singular-value energy rank of alpha-weighted mean k_gj^T U_a on assigned HOLD events",
            "projected_source_message_coordinate_r90": "90% pooled rank of alpha^2-weighted source-message coordinates tilde_mu_gj^T V_a on assigned HOLD events",
            "exact_innovation_write_r90": "90% pooled rank of exact common-span-complement event writes",
            "fitted_innovation_write_r90": "90% pooled rank of the executed V_a-projected fitted event writes",
        },
        "effect_definitions": {
            "promoted": "positive p_exact-p_ablated; the inserted write raises next-token probability",
            "suppressed": "negative p_exact-p_ablated; the inserted write lowers next-token probability",
            "fidelity": "fitted-substitution effect versus exact-addition effect relative to the same exact-write-ablated state",
            "innovation": "only the non-BOS common-span-complement write governed by V_a",
            "complete": "BOS plus common V0 plus unit innovation, with Algorithm 2 reconstructed attention",
        },
        "extraction_audit": audits,
        **results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--exploratory-root", type=Path, default=DEFAULT_EXPLORATORY)
    parser.add_argument("--smoke-report", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
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
    torch.set_num_threads(args.threads)
    result = run(args)
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
