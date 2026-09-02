"""Add readable contexts and render the schema-3 factorized causal replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from transformers import AutoTokenizer


TARGETS = ("L1H1", "L5H2")
VARIANTS = ("message_only", "relation_only", "joint", "complete")


def _fmt(value: float) -> str:
    return f"{value:.2e}" if value != 0 and abs(value) < 1e-4 else f"{value:.3f}"


def _source_list(rows: list[Mapping[str, Any]]) -> str:
    return ", ".join(
        f"{row['token']!r}@{row['position']} ({row['attention']:.3f})" for row in rows
    )


def _token_list(rows: list[Mapping[str, Any]], count: int = 3) -> str:
    return ", ".join(
        f"{row['token']!r} ({row['change']:+.2e})" for row in rows[:count]
    )


def _add_context(payload: dict[str, Any], tokenizer: Any) -> None:
    for target in TARGETS:
        for example in payload[target]["examples"]:
            ids = tokenizer(example["prompt"], add_special_tokens=True)["input_ids"]
            query = int(example["query_position"])
            lo, hi = max(1, query - 40), min(len(ids), query + 13)
            example["context_window"] = (
                ("…" if lo > 1 else "")
                + tokenizer.decode(ids[lo:hi], skip_special_tokens=False)
                + ("…" if hi < len(ids) else "")
            )
            example["context_window_token_bounds"] = [lo, hi]


def _variant(example: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    if name == "complete":
        return example["complete_transport"]["variants"]["complete"]
    return example["factorized_innovation_transport"]["variants"][name]


def _render(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Factorized downstream interpretation of spectral-initialized Algorithm 2",
        "",
        "This development-only replay reloads the closed L1H1 A=2 and L5H2 "
        "A=1 fits at stored ranks r=32,c=64; it performs no refitting. HOLD "
        "events are assigned by Q/K cost alone. At each selected active query, "
        "three substitutions are compared with the same exact-innovation-ablated "
        "state: message-only uses exact attention and P^M_a-projected complement "
        "messages; relation-only uses U_a attention and exact complement messages; "
        "joint uses both fitted parts. The complete substitution instead uses the "
        "exact-complete-head-ablated state, because its fitted write also contains "
        "BOS and the common V_0 component.",
        "",
        "## Audits and assignments",
        "",
        "| head | basis replay | extracted head-output identity | QK/joint agreement (ARI/fraction) | QK semantic ARI | examples | max row-0/prefix logit error |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for target in TARGETS:
        row = payload[target]
        basis = row["basis_audit"]
        agreement = row["qk_vs_joint_holdout_assignment"]
        agreement_text = (
            f"{agreement['ari']:.3f}/{agreement['exact_fraction']:.3f}"
            if row["A"] > 1
            else f"--/{agreement['exact_fraction']:.3f}"
        )
        semantic = f"{row['holdout_qk_semantic_ari']:.3f}" if row["A"] > 1 else "--"
        audits = [row["aggregate_effect_fidelity"][name]["audit"] for name in VARIANTS]
        row0 = max(a["row0_hook_vs_unhooked_logits_max_abs_error"] for a in audits)
        prefix = max(a["causal_prefix_logits_max_abs_error"] for a in audits)
        lines.append(
            f"| {target} | {basis['two_recomputations_max_abs_error']:.1e} | "
            f"{basis['extraction_head_output_max_abs_error']:.1e} | "
            f"{agreement_text} | {semantic} | {len(row['examples'])} | "
            f"{row0:.1e}/{prefix:.1e} |"
        )
    lines += [
        "",
        "All selected prompts retokenize identically to the activation cache. "
        "The row-0 audit compares a hooked but unedited forward with an unhooked "
        "forward; the prefix audit verifies that editing query g changes no logits "
        "at positions before g. L5H2 has only one fitted unit, so its semantic ARI "
        "is not informative.",
        "",
        "## Activated ranks",
        "",
        "Each r90 is the smallest eigenspace of the uncentered pooled activated-"
        "energy second moment containing 90% of its energy. Projected keys use exact "
        "attention and stack "
        "sqrt(alpha_gj/non-BOS-mass_g) k_gj^T U_a; source messages use alpha_gj^2 "
        "weighting from exact attention.",
        "",
        "| split/head/unit | events | stored r/c | q r90 | attention-weighted k r90 | projected source-message r90 | exact/fitted innovation-write r90 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for target in TARGETS:
        row = payload[target]
        for split, key in (("TRAIN", "train_usage_ranks"), ("HOLD", "holdout_qk_usage_ranks")):
            for rank in row[key]:
                lines.append(
                    f"| {split}/{target}/a{rank['unit']} | {rank['events']} | "
                    f"{rank['stored_relation_rank']}/{rank['stored_message_rank']} | "
                    f"{rank['projected_query_r90']} | "
                    f"{rank['attention_weighted_projected_key_r90']} | "
                    f"{rank['attention_squared_projected_source_message_r90']} | "
                    f"{rank['exact_innovation_write_r90']}/"
                    f"{rank['fitted_innovation_write_r90']} |"
                )
    lines += [
        "",
        "## Final-effect fidelity on the 21 inspected HOLD events",
        "",
        "Probability is the signed change in the final next-token distribution. "
        "The transported signature is sqrt(p_exact) times the final log-probability "
        "change. Pooled statistics concatenate vocabulary effects over examples; "
        "macro statistics average the per-example values.",
        "",
        "| head/substitution | probability pooled cos/NMSE | probability macro cos/NMSE | signature pooled cos/NMSE | signature macro cos/NMSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for target in TARGETS:
        for name in VARIANTS:
            row = payload[target]["aggregate_effect_fidelity"][name]
            p, s = row["probability"], row["transported_signature"]
            lines.append(
                f"| {target}/{name.replace('_', '-')} | "
                f"{p['pooled_cosine']:.3f}/{p['pooled_nmse']:.3f} | "
                f"{p['macro_cosine']:.3f}/{p['macro_nmse']:.3f} | "
                f"{s['pooled_cosine']:.3f}/{s['pooled_nmse']:.3f} | "
                f"{s['macro_cosine']:.3f}/{s['macro_nmse']:.3f} |"
            )
    lines += [
        "",
        "### Channel diagnosis",
        "",
        "For L1H1, relation-only is nearly exact in the pooled readout "
        "(probability cosine/NMSE 0.999/0.004; signature 0.997/0.006), whereas "
        "message-only and joint both fail on the innovation effect. Thus the "
        "inspected L1H1 bottleneck is the rank-64 message projector, not U_a's "
        "attention routing. The much better complete-head result includes additional "
        "BOS/common-span components with uncompressed source directions and must not "
        "be read as faithful "
        "innovation recovery.",
        "",
        "For L5H2, each side separately preserves the pooled probability effect "
        "(message-only 0.977/0.161; relation-only 0.995/0.064), and their joint "
        "substitution lowers probability NMSE to 0.032. The signature remains more "
        "message-limited (message-only/joint NMSE 0.284/0.293 versus relation-only "
        "0.107). Macro scores are weaker than pooled scores, so this claim concerns "
        "the aggregate consequential effect rather than uniformly strong fidelity "
        "on every selected event.",
    ]

    for target in TARGETS:
        lines += ["", f"## {target} inspected HOLD events", ""]
        current = None
        for index, example in enumerate(payload[target]["examples"], 1):
            if example["family"] != current:
                current = example["family"]
                lines += [f"### {current.replace('_', ' ')}", ""]
            mass = example["attention_mass"]
            context = example["context_window"].replace("\n", "\n> ")
            joint = _variant(example, "joint")
            exact_next = joint["cached_next_token"]
            lines += [
                f"**Example {index}: source {example['source_id']} "
                f"({example['domain']}), Q/K unit a{example['qk_routed_unit']}.** "
                f"At query {example['query_token']!r}, the cached next token is "
                f"{example['next_token']!r}.",
                "",
                f"> {context}",
                "",
                f"The exact route attends {_source_list(example['exact_top_sources'])}; "
                f"the fitted U route attends {_source_list(example['fitted_top_sources'])}. "
                f"Exact/fitted BOS mass is {mass['exact_bos']:.3f}/"
                f"{mass['fitted_bos']:.3f}; non-BOS mass is "
                f"{mass['exact_nonbos']:.3f}/{mass['fitted_nonbos']:.3f}.",
                "",
                f"The exact innovation promotes {_token_list(joint['exact_promoted_tokens'])} "
                f"and suppresses {_token_list(joint['exact_suppressed_tokens'])}.",
                "",
            ]
            local = example["local_write_fidelity"]
            for name in ("message_only", "relation_only", "joint"):
                effect = _variant(example, name)
                nxt = effect["cached_next_token"]
                lines += [
                    f"The {name.replace('_', '-')} substitution promotes "
                    f"{_token_list(effect['fitted_promoted_tokens'])} and suppresses "
                    f"{_token_list(effect['fitted_suppressed_tokens'])}. Its local "
                    f"write cosine/NMSE is {local[name]['cosine']:.3f}/"
                    f"{local[name]['nmse']:.3f}; its fitted cached-next-token effect "
                    f"is {nxt['fitted_effect']:+.2e} versus the shared exact effect "
                    f"{nxt['exact_effect']:+.2e}.",
                    "",
                ]
            complete = _variant(example, "complete")
            complete_next = complete["cached_next_token"]
            lines += [
                f"For reference, the unedited/innovation-ablated cached-next-token "
                f"probabilities are {_fmt(exact_next['base_probability'])}/"
                f"{_fmt(exact_next['ablated_probability'])}. The complete substitution "
                f"has local cosine/NMSE {local['complete']['cosine']:.3f}/"
                f"{local['complete']['nmse']:.3f} and cached-next-token exact/fitted "
                f"effects {complete_next['exact_effect']:+.2e}/"
                f"{complete_next['fitted_effect']:+.2e}; it promotes "
                f"{_token_list(complete['fitted_promoted_tokens'])} and suppresses "
                f"{_token_list(complete['fitted_suppressed_tokens'])}.",
                "",
            ]

    lines += [
        "## Interpretation boundary",
        "",
        "Message-only isolates compression by P^M_a under exact attention; "
        "relation-only isolates U_a routing while retaining every complement-message "
        "direction; joint exposes their combination and can include nonlinear "
        "interaction through later layers. Complete is not directly comparable to "
        "those three because its ablation reference includes BOS and V_0. These are "
        "development diagnostics of fixed fits, not evidence selecting A or ranks. "
        "No locked or historical test split was read.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    if payload.get("schema_version") != 3:
        raise ValueError("Expected factorized schema 3.")
    if payload.get("locked_test_accessed") or payload.get("historical_test_accessed"):
        raise ValueError("A forbidden split was accessed.")
    expected_counts = {"L1H1": 15, "L5H2": 6}
    for target, expected in expected_counts.items():
        examples = payload[target]["examples"]
        if len(examples) != expected:
            raise ValueError(f"{target}: expected {expected} examples, got {len(examples)}.")
        if not all(row.get("retokenized_prompt_matches_cache") for row in examples):
            raise ValueError(f"{target}: prompt retokenization audit failed.")
        fit_path = Path(payload[target]["fit_artifact"])
        payload[target]["fit_artifact_sha256"] = hashlib.sha256(
            fit_path.read_bytes()
        ).hexdigest()
        for name in VARIANTS:
            aggregate = payload[target]["aggregate_effect_fidelity"][name]
            if any(float(value) != 0.0 for value in aggregate["audit"].values()):
                raise ValueError(f"{target}/{name}: exact logit audit failed.")
            for metric in ("probability", "transported_signature", "log_probability"):
                if not all(math.isfinite(float(value)) for value in aggregate[metric].values()):
                    raise ValueError(f"{target}/{name}/{metric}: non-finite metric.")
    payload["rank_definition"] = (
        "smallest eigenspace carrying 90% of an uncentered pooled activated-energy "
        "second moment; projected keys use exact-attention stacked "
        "sqrt(alpha/non-BOS-mass) rows and source messages use exact alpha^2 weights"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    _add_context(payload, tokenizer)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.output_markdown.write_text(_render(payload) + "\n")


if __name__ == "__main__":
    main()
