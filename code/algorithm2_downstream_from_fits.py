"""Audited Q/K-routed downstream interpretation from saved Algorithm 2 fits.

This script performs no fitting.  It reloads the exact A=2 L1H1 and A=1
L5H2 artifacts produced by ``algorithm2_downstream.py``, re-extracts only the
frozen natural TRAIN/HOLD data, derives Q/K-only HOLD labels, and runs the
causal write substitutions requested for interpretation.
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


def _configure(exploratory: Path, repo_root: Path, here: Path) -> None:
    for path in (
        here,
        exploratory / "code",
        exploratory / "spectral" / "code",
        repo_root,
        repo_root / "unifying_algorithm",
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _covariance_r90(covariance: torch.Tensor) -> int:
    values = torch.linalg.eigvalsh((covariance + covariance.T) / 2).clamp_min(0)
    values = values.flip(0)
    if not len(values) or float(values.sum()) == 0:
        return 0
    return int((values.cumsum(0) < 0.90 * values.sum()).sum()) + 1


def _usage_ranks(
    sources: Sequence[Any],
    labels: Sequence[torch.Tensor],
    relation: Sequence[torch.Tensor],
    message: Sequence[torch.Tensor],
    alg: Any,
) -> list[Mapping[str, Any]]:
    result = []
    for unit, (u, v) in enumerate(zip(relation, message)):
        q_cov = torch.zeros(u.shape[1], u.shape[1], dtype=DTYPE)
        k_cov = torch.zeros_like(q_cov)
        source_cov = torch.zeros(v.shape[1], v.shape[1], dtype=DTYPE)
        exact_cov = torch.zeros(v.shape[0], v.shape[0], dtype=DTYPE)
        fitted_cov = torch.zeros_like(exact_cov)
        count = 0
        for source, local_labels in zip(sources, labels):
            indices = (local_labels == unit).nonzero(as_tuple=True)[0]
            if not len(indices):
                continue
            count += len(indices)
            q_coordinates = source.q.index_select(0, indices) @ u
            q_cov += q_coordinates.T @ q_coordinates

            alpha = source.alpha.index_select(0, indices)[:, 1:]
            mass = alpha.sum(1, keepdim=True).clamp_min(1e-30)
            k_coordinates = source.k[1:] @ u
            # Equivalent to stacking sqrt(alpha_gj/non-BOS-mass_g) k_gj^T U_a.
            k_weight = (alpha / mass).sum(0)
            k_cov += k_coordinates.T @ (k_weight[:, None] * k_coordinates)

            source_coordinates = source.m[1:] @ v
            source_weight = alpha.square().sum(0)
            source_cov += source_coordinates.T @ (
                source_weight[:, None] * source_coordinates
            )

            exact = source.exact_y.index_select(0, indices)
            exact_cov += exact.T @ exact
            alpha_hat = alg._predicted_attention(source, u).index_select(0, indices)
            b = alpha_hat[:, 1:] @ source.m[1:]
            fitted = (b @ v) @ v.T
            fitted_cov += fitted.T @ fitted
        result.append(
            {
                "unit": unit,
                "events": count,
                "stored_relation_rank": int(u.shape[1]),
                "stored_message_rank": int(v.shape[1]),
                "projected_query_r90": _covariance_r90(q_cov),
                "attention_weighted_projected_key_r90": _covariance_r90(k_cov),
                "attention_squared_projected_source_message_r90": _covariance_r90(
                    source_cov
                ),
                "exact_innovation_write_r90": _covariance_r90(exact_cov),
                "fitted_innovation_write_r90": _covariance_r90(fitted_cov),
            }
        )
    return result


def _decoded_tokens(
    values: torch.Tensor, tokenizer: Any, *, largest: bool, count: int
) -> list[Mapping[str, Any]]:
    special = set(getattr(tokenizer, "all_special_ids", ()))
    rows = []
    for token_id in values.argsort(descending=largest).tolist():
        if int(token_id) in special:
            continue
        token = tokenizer.decode([int(token_id)]).replace("\n", "\\n")
        if not token.replace("\\n", "").strip():
            continue
        rows.append({"token": token, "change": float(values[token_id])})
        if len(rows) == count:
            break
    return rows


def _cosine(cross: float, left_sq: float, right_sq: float) -> float:
    return cross / math.sqrt(max(left_sq * right_sq, 1e-30))


@torch.no_grad()
def _transport_variants(
    model: Any,
    tokenizer: Any,
    *,
    ids: torch.Tensor,
    query: int,
    next_token_id: int,
    layer: int,
    exact_write: torch.Tensor,
    fitted_writes: Mapping[str, torch.Tensor],
    reference: str,
) -> Mapping[str, Any]:
    names = tuple(fitted_writes)
    batch = ids[None].expand(2 + len(names), -1).clone()
    module = model.model.layers[layer].self_attn

    def edit(_module: Any, _inputs: Any, output: Any) -> Any:
        values = output[0] if isinstance(output, tuple) else output
        changed = values.clone()
        exact = exact_write.to(changed)
        changed[1, query] -= exact
        for row, name in enumerate(names, start=2):
            changed[row, query] += fitted_writes[name].to(changed) - exact
        return (changed,) + tuple(output[1:]) if isinstance(output, tuple) else changed

    handle = module.register_forward_hook(edit)
    try:
        logits = model(input_ids=batch, use_cache=False).logits.detach().to(DTYPE)
    finally:
        handle.remove()
    plain = model(input_ids=batch[:1], use_cache=False).logits.detach().to(DTYPE)
    row0_error = float((plain[0] - logits[0]).abs().max())
    prefix_error = 0.0
    if query:
        prefix_error = max(
            float((logits[row, :query] - logits[0, :query]).abs().max())
            for row in range(1, len(logits))
        )

    exact_lp = logits[0, query].log_softmax(-1)
    ablated_lp = logits[1, query].log_softmax(-1)
    exact_p, ablated_p = exact_lp.exp(), ablated_lp.exp()
    exact_probability = exact_p - ablated_p
    exact_logprob = exact_lp - ablated_lp
    exact_signature = exact_p.sqrt() * exact_logprob

    def sufficient(exact: torch.Tensor, fitted: torch.Tensor) -> Mapping[str, float]:
        exact_sq = float(exact.square().sum())
        fitted_sq = float(fitted.square().sum())
        cross = float((exact * fitted).sum())
        return {
            "exact_squared_norm": exact_sq,
            "fitted_squared_norm": fitted_sq,
            "cross": cross,
            "squared_error": exact_sq + fitted_sq - 2 * cross,
            "cosine": _cosine(cross, exact_sq, fitted_sq),
            "nmse": (exact_sq + fitted_sq - 2 * cross) / max(exact_sq, 1e-30),
        }

    exact_top = set(torch.topk(exact_probability.abs(), 20).indices.tolist())
    variants = {}
    for row, name in enumerate(names, start=2):
        fitted_lp = logits[row, query].log_softmax(-1)
        fitted_p = fitted_lp.exp()
        fitted_probability = fitted_p - ablated_p
        fitted_logprob = fitted_lp - ablated_lp
        fitted_signature = exact_p.sqrt() * fitted_logprob
        fitted_top = set(torch.topk(fitted_probability.abs(), 20).indices.tolist())
        variants[name] = {
            "probability": sufficient(exact_probability, fitted_probability),
            "transported_signature": sufficient(exact_signature, fitted_signature),
            "log_probability": sufficient(exact_logprob, fitted_logprob),
            "top20_absolute_effect_jaccard": len(exact_top & fitted_top)
            / len(exact_top | fitted_top),
            "cached_next_token": {
                "id": next_token_id,
                "base_probability": float(exact_p[next_token_id]),
                "ablated_probability": float(ablated_p[next_token_id]),
                "fitted_substitution_probability": float(fitted_p[next_token_id]),
                "exact_effect": float(exact_probability[next_token_id]),
                "fitted_effect": float(fitted_probability[next_token_id]),
            },
            "exact_promoted_tokens": _decoded_tokens(
                exact_probability, tokenizer, largest=True, count=6
            ),
            "exact_suppressed_tokens": _decoded_tokens(
                exact_probability, tokenizer, largest=False, count=6
            ),
            "fitted_promoted_tokens": _decoded_tokens(
                fitted_probability, tokenizer, largest=True, count=6
            ),
            "fitted_suppressed_tokens": _decoded_tokens(
                fitted_probability, tokenizer, largest=False, count=6
            ),
        }
    return {
        "reference": reference,
        "variants": variants,
        "audit": {
            "row0_hook_vs_unhooked_logits_max_abs_error": row0_error,
            "causal_prefix_logits_max_abs_error": prefix_error,
        },
    }


def _aggregate(
    examples: Sequence[Mapping[str, Any]], channel: str, variant: str
) -> Mapping[str, Any]:
    transports = [row[channel] for row in examples]
    rows = [transport["variants"][variant] for transport in transports]
    result: dict[str, Any] = {}
    for metric in ("probability", "transported_signature", "log_probability"):
        exact_sq = sum(float(row[metric]["exact_squared_norm"]) for row in rows)
        fitted_sq = sum(float(row[metric]["fitted_squared_norm"]) for row in rows)
        cross = sum(float(row[metric]["cross"]) for row in rows)
        error = exact_sq + fitted_sq - 2 * cross
        result[metric] = {
            "pooled_cosine": _cosine(cross, exact_sq, fitted_sq),
            "pooled_nmse": error / max(exact_sq, 1e-30),
            "macro_cosine": sum(float(row[metric]["cosine"]) for row in rows)
            / max(len(rows), 1),
            "macro_nmse": sum(float(row[metric]["nmse"]) for row in rows)
            / max(len(rows), 1),
            "exact_pooled_norm": math.sqrt(max(exact_sq, 0)),
            "fitted_pooled_norm": math.sqrt(max(fitted_sq, 0)),
        }
    result["top20_absolute_effect_jaccard_macro"] = sum(
        float(row["top20_absolute_effect_jaccard"]) for row in rows
    ) / max(len(rows), 1)
    result["audit"] = {
        key: max(float(row["audit"][key]) for row in transports)
        for key in (
            "row0_hook_vs_unhooked_logits_max_abs_error",
            "causal_prefix_logits_max_abs_error",
        )
    }
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
) -> Mapping[str, Any]:
    exact_alpha = source.alpha[event].to(DTYPE)
    fitted_alpha = alg._predicted_attention(source, relation[unit])[event]
    original = source.base.messages.to(DTYPE)
    exact_innovation_complement = source.exact_y[event]
    b = fitted_alpha[1:] @ source.m[1:]
    message_only_complement = (
        (exact_innovation_complement @ message[unit]) @ message[unit].T
    )
    relation_only_complement = b
    joint_complement = (b @ message[unit]) @ message[unit].T
    exact_innovation_coordinate = exact_innovation_complement @ complement.T
    message_only_coordinate = message_only_complement @ complement.T
    relation_only_coordinate = relation_only_complement @ complement.T
    joint_coordinate = joint_complement @ complement.T
    common_coordinate = (fitted_alpha[1:] @ (original @ v0)[1:]) @ v0.T
    fitted_complete_coordinate = (
        fitted_alpha[0] * original[0]
        + common_coordinate
        + joint_coordinate
    )
    exact_complete_coordinate = exact_alpha @ original

    exact_innovation = exact_innovation_coordinate @ basis.T
    message_only = message_only_coordinate @ basis.T
    relation_only = relation_only_coordinate @ basis.T
    joint = joint_coordinate @ basis.T
    exact_complete = exact_complete_coordinate @ basis.T
    fitted_complete = fitted_complete_coordinate @ basis.T

    def local(exact: torch.Tensor, fitted: torch.Tensor) -> Mapping[str, float]:
        exact_sq = float(exact.square().sum())
        fitted_sq = float(fitted.square().sum())
        cross = float((exact * fitted).sum())
        return {
            "exact_norm": math.sqrt(exact_sq),
            "fitted_norm": math.sqrt(fitted_sq),
            "cosine": _cosine(cross, exact_sq, fitted_sq),
            "nmse": (exact_sq + fitted_sq - 2 * cross) / max(exact_sq, 1e-30),
        }
    return {
        "exact_innovation": exact_innovation,
        "message_only": message_only,
        "relation_only": relation_only,
        "joint": joint,
        "exact_complete": exact_complete,
        "fitted_complete": fitted_complete,
        "local_write_fidelity": {
            "message_only": local(exact_innovation, message_only),
            "relation_only": local(exact_innovation, relation_only),
            "joint": local(exact_innovation, joint),
            "complete": local(exact_complete, fitted_complete),
        },
        "attention_mass": {
            "exact_bos": float(exact_alpha[0]),
            "fitted_bos": float(fitted_alpha[0]),
            "exact_nonbos": float(exact_alpha[1:].sum()),
            "fitted_nonbos": float(fitted_alpha[1:].sum()),
        },
    }


def _run_target(
    model: Any,
    tokenizer: Any,
    corpus: Mapping[str, Any],
    raw_sources: Sequence[Any],
    extraction_audit: Mapping[str, Any],
    *,
    target: str,
    layer: int,
    head: int,
    fit_path: Path,
    alg: Any,
    helper: Any,
    compute_message_basis: Any,
    head_output_block: Any,
) -> Mapping[str, Any]:
    v0, complement, c0 = alg._fit_common_span(raw_sources)
    prepared = alg._prepare_sources(raw_sources)
    train = [row for row in prepared if row.base.split == "train"]
    holdout = [row for row in prepared if row.base.split == "holdout"]
    fit = torch.load(fit_path, map_location="cpu", weights_only=False)
    relation = tuple(row.to(DTYPE) for row in fit["relation"])
    message = tuple(row.to(DTYPE) for row in fit["message"])
    train_labels = tuple(row.to(torch.long) for row in fit["train_assignments"])
    if float((fit["v0"] - v0).abs().max()) > 1e-12:
        raise RuntimeError(f"{target} common span differs from saved fit.")
    if float((fit["complement"] - complement).abs().max()) > 1e-12:
        raise RuntimeError(f"{target} complement differs from saved fit.")

    block = head_output_block(model, layer_idx=layer, head=head)
    basis1 = compute_message_basis(block, expected_rank=256).basis
    basis2 = compute_message_basis(block, expected_rank=256).basis
    basis_recompute_error = float((basis1 - basis2).abs().max())
    if basis_recompute_error > 1e-12:
        raise RuntimeError(f"{target} deterministic message basis did not replay.")
    head_output_error = float(extraction_audit.get("head_output_max_abs_error", math.inf))
    if head_output_error != 0.0:
        raise RuntimeError(f"{target} exact extracted head-output identity failed.")

    scales = fit["scales"]
    joint_labels, _ = alg._event_costs(
        holdout, relation, message, scales, qk_only=False
    )
    qk_labels, _ = alg._event_costs(
        holdout, relation, message, scales, qk_only=True
    )
    joint_flat, qk_flat = torch.cat(joint_labels), torch.cat(qk_labels)
    label_agreement = {
        "exact_fraction": float((joint_flat == qk_flat).to(DTYPE).mean()),
        "ari": alg._ari(joint_flat, qk_flat),
        "permutation_accuracy": alg._permutation_accuracy(joint_flat, qk_flat),
    }

    source_text = {str(row["source_id"]): str(row["text"]) for row in corpus["sources"]}
    selected = helper._choose_examples(holdout, qk_labels, target=target)
    examples = []
    for source_index, event, unit in selected:
        source = holdout[source_index]
        text = source_text[str(source.base.source_id)]
        ids = torch.tensor(tokenizer(text, add_special_tokens=True)["input_ids"])
        retokens = tuple(
            tokenizer.decode([int(token)], skip_special_tokens=False)
            for token in ids.tolist()
        )
        if retokens != source.base.tokens:
            raise RuntimeError("Retokenized prompt differs from cached source.")
        query = int(source.base.query_positions[event])
        if query + 1 >= len(ids):
            raise RuntimeError("Selected event lacks a next-token target.")
        writes = _event_writes(
            source,
            event,
            int(unit),
            relation=relation,
            message=message,
            v0=v0,
            complement=complement,
            basis=basis1,
            alg=alg,
        )
        innovation = _transport_variants(
            model,
            tokenizer,
            ids=ids,
            query=query,
            next_token_id=int(ids[query + 1]),
            layer=layer,
            exact_write=writes["exact_innovation"],
            fitted_writes={
                "message_only": writes["message_only"],
                "relation_only": writes["relation_only"],
                "joint": writes["joint"],
            },
            reference="exact_innovation_ablation",
        )
        complete = _transport_variants(
            model,
            tokenizer,
            ids=ids,
            query=query,
            next_token_id=int(ids[query + 1]),
            layer=layer,
            exact_write=writes["exact_complete"],
            fitted_writes={"complete": writes["fitted_complete"]},
            reference="exact_complete_head_ablation",
        )
        examples.append(
            {
                "source_id": str(source.base.source_id),
                "domain": source.base.domain,
                "family": source.base.families[event],
                "qk_routed_unit": int(unit),
                "joint_routed_unit": int(joint_labels[source_index][event]),
                "prompt": text,
                "query_position": query,
                "query_token": source.base.tokens[query].replace("\n", "\\n"),
                "next_token": source.base.tokens[query + 1].replace("\n", "\\n"),
                "exact_top_sources": helper._top_sources(source, event),
                "fitted_top_sources": helper._top_predicted_sources(
                    source, event, relation[int(unit)], alg
                ),
                "attention_mass": writes["attention_mass"],
                "local_write_fidelity": writes["local_write_fidelity"],
                "factorized_innovation_transport": innovation,
                "complete_transport": complete,
                "retokenized_prompt_matches_cache": True,
            }
        )

    composition = []
    for unit in range(len(relation)):
        counts: dict[str, int] = defaultdict(int)
        for source, labels in zip(holdout, qk_labels):
            for family, label in zip(source.base.families, labels.tolist()):
                if int(label) == unit:
                    counts[family] += 1
        composition.append({"unit": unit, "family_counts": dict(sorted(counts.items()))})

    return {
        "target": target,
        "layer": layer,
        "head": head,
        "A": len(relation),
        "common_rank": c0,
        "basis_audit": {
            "two_recomputations_max_abs_error": basis_recompute_error,
            "extraction_head_output_max_abs_error": head_output_error,
        },
        "qk_vs_joint_holdout_assignment": label_agreement,
        "holdout_qk_semantic_ari": alg._ari(
            qk_flat, torch.cat([row.base.semantic_labels for row in holdout])
        ),
        "train_usage_ranks": _usage_ranks(
            train, train_labels, relation, message, alg
        ),
        "holdout_qk_usage_ranks": _usage_ranks(
            holdout, qk_labels, relation, message, alg
        ),
        "holdout_qk_unit_composition": composition,
        "aggregate_effect_fidelity": {
            "message_only": _aggregate(
                examples, "factorized_innovation_transport", "message_only"
            ),
            "relation_only": _aggregate(
                examples, "factorized_innovation_transport", "relation_only"
            ),
            "joint": _aggregate(
                examples, "factorized_innovation_transport", "joint"
            ),
            "complete": _aggregate(examples, "complete_transport", "complete"),
        },
        "examples": examples,
        "fit_artifact": str(fit_path),
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    here = Path(__file__).resolve().parent
    _configure(args.exploratory_root.resolve(), args.repo_root.resolve(), here)
    import algorithm1
    import algorithm2 as alg
    import algorithm2_downstream as helper
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
    model, tokenizer = load_registered_model_and_tokenizer(artifacts, allow_download=False)
    model.to(device="cpu", dtype=torch.float32).eval()
    result = {}
    audits = {}
    try:
        for target, layer, head in (("L1H1", 1, 1), ("L5H2", 5, 2)):
            sources, audit = algorithm1._extract_head(
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
            result[target] = _run_target(
                model,
                tokenizer,
                corpus,
                sources,
                audit,
                target=target,
                layer=layer,
                head=head,
                fit_path=args.artifact_dir / f"{target}_fit.pt",
                alg=alg,
                helper=helper,
                compute_message_basis=compute_message_basis,
                head_output_block=head_output_block,
            )
    finally:
        del model
        del tokenizer

    return {
        "schema_version": 3,
        "analysis": "algorithm2_spectral_qk_routed_downstream_factorized_from_saved_fits",
        "development_only": True,
        "locked_test_accessed": False,
        "historical_test_accessed": False,
        "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        "configuration": {
            "L1H1": {"A": 2, "r": 32, "c": 64},
            "L5H2": {"A": 1, "r": 32, "c": 64},
            "holdout_assignment": "Q/K-only argmin_a R_g,a",
            "intervention": {
                "message_only": "exact attention with the selected P^M_a-projected complement messages, versus exact-innovation ablation",
                "relation_only": "selected U_a attention with exact complement messages, versus exact-innovation ablation",
                "joint": "selected U_a attention with selected P^M_a-projected complement messages, versus exact-innovation ablation",
                "complete": "fitted BOS, common, relation, and message write, versus exact complete-head ablation",
            },
            "readout": "final next-token distribution at active query",
        },
        "rank_definition": "smallest covariance eigenspace carrying 90% pooled activated-coordinate energy; projected keys use stacked sqrt(alpha/non-BOS-mass) rows",
        "extraction_audit": audits,
        **result,
    }


def main() -> None:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--exploratory-root", type=Path, default=default_root)
    parser.add_argument("--smoke-report", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    payload = run(args)
    payload["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
