"""Freeze a source-disjoint natural corpus for IT L1H1 and L5H2.

This is a pre-fitting data-construction step. Candidate events are specified
from tokens alone; exact head activity is used only for the common strong-event
gate. No attention-unit estimator or model selection is run.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "eda"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eda._paths import configure_unifying_attention

configure_unifying_attention()

from eda.headvis_natural_mechanism_audit import (
    DOMAINS,
    _choose_sources,
    _run_head,
    l1_rule_events,
    l17_rule_events,
    source_split,
)
from eda.natural_headvis_prompt_corpus import _source_from_tokens
from unifying_attention.experiments.unlearned_projector_real import (
    load_registered_model_and_tokenizer,
    resolve_registered_snapshot,
)
from unifying_attention.unlearned_projector_gate import require_valid_smoke_report


SCHEMA_VERSION = 1
L1_FAMILIES = frozenset(
    {
        "alphabetic_word",
        "four_digit_year",
        "identifier",
        "newline_boundary",
        "other_number",
    }
)
L5_FAMILIES = frozenset({"strict_induction", "tokenization_tolerant_induction"})


def _corpus_split(value: str) -> str:
    if value == "discovery":
        return "train"
    if value == "confirmation":
        return "holdout"
    raise ValueError(f"Unknown audit split: {value}")


def _rename_split(record: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(record)
    if "split" in result:
        result["split"] = _corpus_split(str(result["split"]))
    return result


def _counts(records: Sequence[Mapping[str, Any]]) -> Mapping[str, int]:
    counts = Counter((row["family"], row["split"]) for row in records)
    return {"|".join(key): value for key, value in sorted(counts.items())}


def _source_counts(records: Sequence[Mapping[str, Any]]) -> Mapping[str, int]:
    grouped: dict[tuple[str, str], set[str]] = {}
    for row in records:
        grouped.setdefault((str(row["family"]), str(row["split"])), set()).add(
            str(row["source_id"])
        )
    return {"|".join(key): len(value) for key, value in sorted(grouped.items())}


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    require_valid_smoke_report(Path(args.smoke_report).resolve())
    sequences_path = Path(args.headvis_sequences).resolve()
    sequences = json.loads(sequences_path.read_text())
    artifacts = resolve_registered_snapshot(allow_download=False)
    model, tokenizer = load_registered_model_and_tokenizer(artifacts, allow_download=False)
    try:
        all_sources = [
            _source_from_tokens(tokenizer, str(source_id), exported)
            for source_id, exported in sequences.items()
        ]
        sources = _choose_sources(
            all_sources, per_domain_split=args.sources_per_domain_split
        )
        l1_events = {
            source.source_id: tuple(
                event
                for event in l1_rule_events(source, maximum_per_family=10)
                if event.family in L1_FAMILIES
            )
            for source in sources
        }
        l5_events = {
            source.source_id: tuple(
                event
                for event in l17_rule_events(source)
                if event.family in L5_FAMILIES
            )
            for source in sources
        }
        l1 = _run_head(
            model,
            tokenizer,
            sources,
            l1_events,
            layer=1,
            head=1,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            inventory_routes=False,
            include_strong_manifest=True,
        )
        l5 = _run_head(
            model,
            tokenizer,
            sources,
            l5_events,
            layer=5,
            head=2,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            inventory_routes=False,
            include_strong_manifest=True,
        )
    finally:
        del model
        del tokenizer

    l1_records = [_rename_split(row) for row in l1.pop("strong_event_manifest")]
    l5_records = [_rename_split(row) for row in l5.pop("strong_event_manifest")]
    source_rows = [
        {
            "source_id": source.source_id,
            "domain": source.domain,
            "split": _corpus_split(source_split(source.source_id)),
            "text": source.text,
            "text_sha256": hashlib.sha256(source.text.encode()).hexdigest(),
        }
        for source in sources
    ]
    train_ids = {row["source_id"] for row in source_rows if row["split"] == "train"}
    holdout_ids = {row["source_id"] for row in source_rows if row["split"] == "holdout"}
    if train_ids & holdout_ids:
        raise RuntimeError("Natural corpus source split leaked.")
    for head, records in (("L1H1", l1_records), ("L5H2", l5_records)):
        if any(
            (row["source_id"] in train_ids) != (row["split"] == "train")
            for row in records
        ):
            raise RuntimeError(f"{head} event split disagrees with its source split.")

    return {
        "schema_version": SCHEMA_VERSION,
        "analysis": "freeze_natural_behavior_corpus",
        "development_only": True,
        "attention_unit_fit": False,
        "model_selection_run": False,
        "historical_test_accessed": False,
        "locked_test_accessed": False,
        "model": {
            "identifier": "google/gemma-3-1b-it",
            "revision": artifacts.revision,
            "model_sha256": artifacts.model_sha256,
            "tokenizer_sha256": artifacts.tokenizer_sha256,
        },
        "source_asset": {
            "path_used": str(sequences_path),
            "sha256": hashlib.sha256(sequences_path.read_bytes()).hexdigest(),
            "source_documents": len(sequences),
        },
        "sampling": {
            "sources": len(sources),
            "sources_per_domain_split": args.sources_per_domain_split,
            "domains": list(DOMAINS),
            "split": "SHA256 natural-mechanism-split(source_id) parity",
            "source_disjoint": True,
            "candidate_enumeration_reads_head_activity": False,
            "maximum_events_per_source_family": {"L1H1": 10, "L5H2": 5},
            "strong_gate": {
                "target_rank": 1,
                "minimum_attention": 0.10,
                "minimum_pair_write_energy": 0.10,
            },
            "bos_policy": (
                "BOS remains in every complete causal attention row but is never a "
                "semantic target or label; admitted events require a non-BOS rule "
                "target to win the strong gate."
            ),
        },
        "sources": source_rows,
        "L1H1": {
            "candidate_function_hypothesis": (
                "surface composition versus newline/structural-boundary routing"
            ),
            "rule_families": sorted(L1_FAMILIES),
            "strong_events_by_family_split": _counts(l1_records),
            "sources_by_family_split": _source_counts(l1_records),
            "extraction_audit": l1["extraction_audit"],
            "elapsed_seconds": l1["elapsed_seconds"],
            "events": l1_records,
        },
        "L5H2": {
            "candidate_function_hypothesis": (
                "one induction/continuation-selection operation with exact and "
                "tokenization-tolerant address variants"
            ),
            "rule_families": sorted(L5_FAMILIES),
            "strong_events_by_family_split": _counts(l5_records),
            "sources_by_family_split": _source_counts(l5_records),
            "extraction_audit": l5["extraction_audit"],
            "elapsed_seconds": l5["elapsed_seconds"],
            "events": l5_records,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-report", type=Path, required=True)
    parser.add_argument("--headvis-sequences", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources-per-domain-split", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if min(
        args.sources_per_domain_split,
        args.chunk_size,
        args.batch_size,
        args.threads,
    ) <= 0:
        raise SystemExit("All numeric controls must be positive.")
    torch.set_num_threads(args.threads)
    result = run(args)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
