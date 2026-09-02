"""Build a diverse natural prompt corpus for Gemma IT L1H3 and L17H0.

The source texts are the public Gemma-3-1B-PT HeadVis standard-distribution
sequences.  They are retokenized exactly with the pinned Gemma-3-1B-IT
tokenizer and all activations are recomputed on IT.  The script validates head
biology and writes a development manifest; it does not fit attention units.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "eda"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eda._paths import configure_unifying_attention

configure_unifying_attention()

from unifying_attention.gemma import extract_layer_qk
from unifying_attention.prompts import EventSpec, PromptSpec, pad_prompts
from unifying_attention.unlearned_projector_gate import require_valid_smoke_report
from unifying_attention.experiments.unlearned_projector_real import (
    load_registered_model_and_tokenizer,
    resolve_registered_snapshot,
)


SCHEMA_VERSION = 1
DOMAINS = ("code", "dialogue", "multilingual", "prose", "structured")
LINE_FAMILIES = ("newline", "word_split", "year_split")
STRONG_ATTENTION = 0.10
STRONG_ENERGY = 0.10
STOPWORDS = frozenset(
    {
        "also", "and", "are", "but", "could", "for", "from", "has",
        "have", "her", "here", "his", "into", "its", "not", "should",
        "than", "that", "the", "their", "then", "there", "they", "this",
        "was", "were", "what", "when", "where", "which", "with", "would",
        "you",
    }
)


@dataclass(frozen=True)
class Candidate:
    source_id: str
    domain: str
    split: str
    family: str
    query: int
    targets: tuple[tuple[int, int], ...]
    annotation: str


@dataclass(frozen=True)
class TokenizedSource:
    source_id: str
    text: str
    input_ids: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]
    tokens: tuple[str, ...]
    domain: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_digest(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(str(part) for part in parts).encode()).hexdigest()


def stable_split(source_id: str) -> str:
    return "holdout" if int(_stable_digest("split", source_id)[:8], 16) % 3 == 0 else "train"


def classify_domain(text: str) -> str:
    lower = text.lower()
    letters = [character for character in text if character.isalpha()]
    nonascii = sum(ord(character) > 127 for character in letters) / max(1, len(letters))
    lines = text.splitlines()
    nonempty = [line for line in lines if line.strip()]
    if nonascii > 0.08:
        return "multilingual"
    if any(marker in lower for marker in ("human:", "assistant:", "q:", "a:")):
        return "dialogue"
    if sum(line.count('"') for line in lines) >= 12:
        return "dialogue"
    code_score = sum(
        lower.count(marker)
        for marker in ("def ", "function ", "import ", "class ", "return ", "</", "/>", "{", "};", " = ")
    )
    if code_score >= 3:
        return "code"
    structured = sum(
        any(delimiter in line for delimiter in ("|", "\t", ":", "="))
        for line in nonempty
    ) / max(1, len(nonempty))
    if len(nonempty) >= 8 and structured > 0.35:
        return "structured"
    return "prose"


def strong_event(record: Mapping[str, Any]) -> bool:
    return bool(
        record["best_target_token_rank"] == 1
        and record["target_attention_mass"] >= STRONG_ATTENTION
        and record["target_pair_write_energy_fraction"] >= STRONG_ENERGY
    )


def _source_from_tokens(tokenizer: Any, source_id: str, exported: Sequence[str]) -> TokenizedSource:
    pieces = tuple(str(value) for value in exported)
    has_bos = bool(pieces and pieces[0] == "<bos>")
    text = "".join(pieces[1:] if has_bos else pieces)
    encoded = tokenizer(text, add_special_tokens=has_bos, return_offsets_mapping=True)
    ids = tuple(int(value) for value in encoded["input_ids"])
    decoded = tuple(tokenizer.decode([value], skip_special_tokens=False) for value in ids)
    # PT and IT tokenizer boundaries can differ around malformed Unicode while
    # still decoding to exactly the same text.  This corpus intentionally
    # retokenizes for IT, so require text preservation rather than PT boundary
    # preservation.
    if "".join(decoded) != "".join(pieces):
        raise RuntimeError(f"HeadVis source {source_id} does not round-trip losslessly.")
    return TokenizedSource(
        source_id=source_id,
        text=text,
        input_ids=ids,
        offsets=tuple((int(start), int(end)) for start, end in encoded["offset_mapping"]),
        tokens=decoded,
        domain=classify_domain(text),
    )


def _overlap_positions(
    offsets: Sequence[tuple[int, int]], start: int, end: int
) -> list[int]:
    return [
        index
        for index, (left, right) in enumerate(offsets)
        if right > start and left < end
    ]


def _choose_one(rows: Sequence[Candidate], *key: object) -> Candidate:
    ordered = sorted(rows, key=lambda row: _stable_digest(*key, row.query, row.annotation))
    return ordered[0]


def line_candidates(source: TokenizedSource) -> Mapping[str, Candidate]:
    by_family: dict[str, list[Candidate]] = defaultdict(list)
    split = stable_split(source.source_id)
    for match in re.finditer(r"(?<![\w])\d{4}(?!\d)", source.text):
        positions = _overlap_positions(source.offsets, match.start(), match.end())
        digit_positions = [
            position
            for position in positions
            if source.tokens[position].strip().isdigit()
        ]
        if len(digit_positions) >= 2:
            query = digit_positions[-1]
            by_family["year_split"].append(
                Candidate(
                    source.source_id,
                    source.domain,
                    split,
                    "year_split",
                    query,
                    ((digit_positions[0], query),),
                    match.group(),
                )
            )
    for match in re.finditer(r"[^\W\d_]{6,}", source.text, re.UNICODE):
        positions = _overlap_positions(source.offsets, match.start(), match.end())
        if len(positions) >= 2:
            query = positions[-1]
            by_family["word_split"].append(
                Candidate(
                    source.source_id,
                    source.domain,
                    split,
                    "word_split",
                    query,
                    ((positions[0], query),),
                    match.group(),
                )
            )
    newlines = [
        index for index, token in enumerate(source.tokens) if "\n" in token
    ]
    for previous, query in zip(newlines, newlines[1:]):
        if query > previous + 1:
            by_family["newline"].append(
                Candidate(
                    source.source_id,
                    source.domain,
                    split,
                    "newline",
                    query,
                    ((previous, previous + 1),),
                    "newline-to-previous-newline",
                )
            )
    return {
        family: _choose_one(rows, source.source_id, family)
        for family, rows in by_family.items()
    }


def copy_candidate(source: TokenizedSource) -> Candidate | None:
    prior: dict[int, list[int]] = defaultdict(list)
    rows: list[Candidate] = []
    split = stable_split(source.source_id)
    for query, current in enumerate(source.input_ids[:-1]):
        prior[current].append(query)
        target_id = source.input_ids[query + 1]
        token = source.tokens[query + 1].strip().lower()
        cleaned = "".join(character for character in token if character.isalnum())
        matches = [position for position in prior.get(target_id, ()) if query - position >= 5]
        if len(cleaned) < 3 or cleaned in STOPWORDS or not matches:
            continue
        rows.append(
            Candidate(
                source.source_id,
                source.domain,
                split,
                "natural_copy_selection",
                query,
                tuple((position, position + 1) for position in matches[-8:]),
                token,
            )
        )
    return _choose_one(rows, source.source_id, "natural_copy_selection") if rows else None


def _select_candidates(
    rows: Iterable[Candidate], *, train_per_stratum: int, holdout_per_stratum: int
) -> tuple[Candidate, ...]:
    grouped: dict[tuple[str, str, str], list[Candidate]] = defaultdict(list)
    for row in rows:
        grouped[(row.family, row.domain, row.split)].append(row)
    chosen = []
    for key, values in sorted(grouped.items()):
        limit = train_per_stratum if key[2] == "train" else holdout_per_stratum
        values.sort(key=lambda row: _stable_digest("candidate", row.source_id, row.family))
        chosen.extend(values[:limit])
    return tuple(chosen)


def _prompts_from_candidates(
    sources: Mapping[str, TokenizedSource], candidates: Sequence[Candidate], *, target: str
) -> tuple[tuple[PromptSpec, ...], Mapping[str, Mapping[str, Any]]]:
    grouped: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.source_id].append(candidate)
    prompts = []
    metadata = {}
    for source_id, rows in sorted(grouped.items(), key=lambda item: int(item[0])):
        source = sources[source_id]
        rows.sort(key=lambda row: (row.query, row.family))
        group_id = f"headvis-gemma3:{source_id}"
        prompts.append(
            PromptSpec(
                target=target,
                input_ids=source.input_ids,
                events=tuple(
                    EventSpec(row.family, row.query, row.targets, ()) for row in rows
                ),
                text=source.text,
                group_id=group_id,
            )
        )
        metadata[group_id] = {
            "source_id": source_id,
            "domain": source.domain,
            "split": stable_split(source_id),
        }
    return tuple(prompts), metadata


def _event_records(
    data: Any,
    prompts: Sequence[PromptSpec],
    metadata: Mapping[str, Mapping[str, Any]],
    tokenizer: Any,
    *,
    head: int,
) -> list[Mapping[str, Any]]:
    records = []
    for prompt_index, prompt in enumerate(prompts):
        meta = metadata[prompt.group_id]
        for event_index, event in enumerate(prompt.events):
            query = int(event.query_pos)
            causal = data.source_mask[prompt_index, query].nonzero().flatten()
            attention = data.attention[prompt_index, head, query, causal].to(torch.float64)
            messages = data.messages[prompt_index, head, causal].to(torch.float64)
            energy = attention.square() * messages.square().sum(1)
            target_positions = {
                position
                for start, end in event.target_spans
                for position in range(start, end)
            }
            target_mask = torch.tensor(
                [int(position) in target_positions for position in causal], dtype=torch.bool
            )
            if not bool(target_mask.any()):
                raise RuntimeError("Annotated target is absent from the causal prefix.")
            target_attention = float(attention[target_mask].sum())
            target_energy = float(energy[target_mask].sum() / energy.sum().clamp_min(1e-30))
            best_target = float(attention[target_mask].max())
            rank = 1 + int((attention > best_target).sum())
            top = attention.topk(min(5, len(attention)))
            top_routes = [
                {
                    "position": int(causal[int(local)]),
                    "token": tokenizer.decode(
                        [prompt.input_ids[int(causal[int(local)])]], skip_special_tokens=False
                    ),
                    "attention": float(value),
                    "is_target": int(causal[int(local)]) in target_positions,
                }
                for value, local in zip(top.values, top.indices, strict=True)
            ]
            record = {
                "event_id": f"{prompt.group_id}:{event_index}:{event.family}:{query}",
                "group_id": prompt.group_id,
                "source_id": meta["source_id"],
                "domain": meta["domain"],
                "split": meta["split"],
                "family": event.family,
                "sequence_length": len(prompt.input_ids),
                "query_position": query,
                "query_fraction": query / max(1, len(prompt.input_ids) - 1),
                "minimum_target_distance": query - max(target_positions),
                "query_token": tokenizer.decode(
                    [prompt.input_ids[query]], skip_special_tokens=False
                ),
                "target_tokens": [
                    tokenizer.decode([prompt.input_ids[position]], skip_special_tokens=False)
                    for position in sorted(target_positions)
                ],
                "target_attention_mass": target_attention,
                "target_pair_write_energy_fraction": target_energy,
                "best_target_token_rank": rank,
                "target_is_top1": rank == 1,
                "top_routes": top_routes,
                "prompt_sha256": hashlib.sha256(prompt.text.encode()).hexdigest(),
                "excerpt": tokenizer.decode(
                    prompt.input_ids[max(0, query - 24) : min(len(prompt.input_ids), query + 12)]
                ),
            }
            record["strong_behavior_gate"] = strong_event(record)
            records.append(record)
    return records


def _extract(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[PromptSpec],
    metadata: Mapping[str, Mapping[str, Any]],
    *,
    layer: int,
    head: int,
    batch_size: int,
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any], float]:
    started = time.perf_counter()
    input_ids, lengths = pad_prompts(prompts, tokenizer)
    data, audit = extract_layer_qk(
        model,
        input_ids=input_ids,
        prompt_lengths=lengths,
        layer_idx=layer,
        batch_size=batch_size,
        recompute_ov_float64=True,
    )
    records = _event_records(data, prompts, metadata, tokenizer, head=head)
    return records, audit, time.perf_counter() - started


def _select_strong(
    records: Sequence[Mapping[str, Any]], *, train_per_stratum: int, holdout_per_stratum: int
) -> tuple[Mapping[str, Any], ...]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if record["strong_behavior_gate"]:
            grouped[(record["family"], record["domain"], record["split"])].append(record)
    selected = []
    for key, values in sorted(grouped.items()):
        limit = train_per_stratum if key[2] == "train" else holdout_per_stratum
        values.sort(key=lambda row: _stable_digest("strong", row["event_id"]))
        selected.extend(values[:limit])
    return tuple(selected)


def _summary(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["family"], record["domain"], record["split"])].append(record)
    return {
        "by_family_domain_split": {
            "|".join(key): {
                "events": len(values),
                "strong_events": sum(bool(row["strong_behavior_gate"]) for row in values),
                "mean_target_attention_mass": sum(row["target_attention_mass"] for row in values) / len(values),
                "mean_target_pair_write_energy_fraction": sum(
                    row["target_pair_write_energy_fraction"] for row in values
                ) / len(values),
                "target_top1_rate": sum(bool(row["target_is_top1"]) for row in values) / len(values),
            }
            for key, values in sorted(grouped.items())
        },
        "total_events": len(records),
        "strong_events": sum(bool(record["strong_behavior_gate"]) for record in records),
        "unique_sources": len({record["source_id"] for record in records}),
    }


def _manifest(
    selected: Sequence[Mapping[str, Any]],
    prompts: Sequence[PromptSpec],
    metadata: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    selected_by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in selected:
        selected_by_group[record["group_id"]].append(record)
    rows = []
    prompt_by_group = {prompt.group_id: prompt for prompt in prompts}
    for group_id, events in sorted(selected_by_group.items()):
        prompt = prompt_by_group[group_id]
        rows.append(
            {
                **metadata[group_id],
                "group_id": group_id,
                "text": prompt.text,
                "prompt_sha256": hashlib.sha256(prompt.text.encode()).hexdigest(),
                "selected_events": sorted(events, key=lambda row: row["event_id"]),
            }
        )
    return {
        "prompts": rows,
        "prompt_count": len(rows),
        "event_count": len(selected),
        "train_source_ids": sorted(
            {row["source_id"] for row in rows if row["split"] == "train"}, key=int
        ),
        "holdout_source_ids": sorted(
            {row["source_id"] for row in rows if row["split"] == "holdout"}, key=int
        ),
    }


def _pt_same_index_audit(
    sequences: Mapping[str, Sequence[str]], l1_umap: Mapping[str, Any], l17_umap: Mapping[str, Any]
) -> Mapping[str, Any]:
    def classify(umap: Mapping[str, Any]) -> Mapping[str, int]:
        counts = Counter()
        for source_id, query, key in zip(umap["seq_idx"], umap["top_q"], umap["top_k"]):
            tokens = sequences[str(source_id)]
            segment = "".join(tokens[int(key) : int(query) + 1]).strip()
            key_token = str(tokens[int(key)])
            query_token = str(tokens[int(query)])
            if int(key) < int(query) and segment.isdigit() and len(segment) == 4:
                counts["year_split"] += 1
            elif int(key) < int(query) and "\n" in key_token and "\n" in query_token:
                counts["newline"] += 1
            elif (
                int(key) < int(query)
                and key_token.strip().isalpha()
                and query_token.strip().isalpha()
                and segment.isalpha()
            ):
                counts["word_split"] += 1
            else:
                counts["other"] += 1
        return dict(counts)

    return {
        "checkpoint": "google/gemma-3-1b-pt",
        "L1H3_top_3000_pair_rules": classify(l1_umap),
        "L17H0_top_3000_pair_rules": classify(l17_umap),
        "decision": "Do not transfer the IT biological labels to the same PT head indices.",
    }


def run(args: argparse.Namespace) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    smoke = Path(args.smoke_report).resolve()
    require_valid_smoke_report(smoke)
    sequences_path = Path(args.headvis_sequences).resolve()
    l1_umap_path = Path(args.pt_l1_umap).resolve()
    l17_umap_path = Path(args.pt_l17_umap).resolve()
    sequences = json.loads(sequences_path.read_text())
    l1_umap = json.loads(l1_umap_path.read_text())
    l17_umap = json.loads(l17_umap_path.read_text())
    artifacts = resolve_registered_snapshot(allow_download=False)
    model, tokenizer = load_registered_model_and_tokenizer(artifacts, allow_download=False)
    try:
        sources: dict[str, TokenizedSource] = {}
        line_rows: list[Candidate] = []
        copy_rows: list[Candidate] = []
        for source_id, exported in sequences.items():
            source = _source_from_tokens(tokenizer, str(source_id), exported)
            sources[str(source_id)] = source
            line_rows.extend(line_candidates(source).values())
            copied = copy_candidate(source)
            if copied is not None:
                copy_rows.append(copied)
        line_chosen = _select_candidates(
            line_rows,
            train_per_stratum=args.line_candidate_train,
            holdout_per_stratum=args.line_candidate_holdout,
        )
        copy_chosen = _select_candidates(
            copy_rows,
            train_per_stratum=args.copy_candidate_train,
            holdout_per_stratum=args.copy_candidate_holdout,
        )
        line_prompts, line_metadata = _prompts_from_candidates(
            sources, line_chosen, target="natural-line-width"
        )
        copy_prompts, copy_metadata = _prompts_from_candidates(
            sources, copy_chosen, target="natural-copy-selection"
        )
        line_records, line_audit, line_seconds = _extract(
            model,
            tokenizer,
            line_prompts,
            line_metadata,
            layer=1,
            head=3,
            batch_size=args.batch_size,
        )
        copy_records, copy_audit, copy_seconds = _extract(
            model,
            tokenizer,
            copy_prompts,
            copy_metadata,
            layer=17,
            head=0,
            batch_size=args.batch_size,
        )
    finally:
        del model
        del tokenizer
    selected_line = _select_strong(
        line_records,
        train_per_stratum=args.line_final_train,
        holdout_per_stratum=args.line_final_holdout,
    )
    selected_copy = _select_strong(
        copy_records,
        train_per_stratum=args.copy_final_train,
        holdout_per_stratum=args.copy_final_holdout,
    )
    line_manifest = _manifest(selected_line, line_prompts, line_metadata)
    copy_manifest = _manifest(selected_copy, copy_prompts, copy_metadata)
    if set(line_manifest["train_source_ids"]) & set(line_manifest["holdout_source_ids"]):
        raise RuntimeError("Line-width source leaked across splits.")
    if set(copy_manifest["train_source_ids"]) & set(copy_manifest["holdout_source_ids"]):
        raise RuntimeError("Copy-selection source leaked across splits.")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "development_only": True,
        "source": "Gemma-3-1B-PT HeadVis standard-distribution sequences",
        "source_sha256": _sha256(sequences_path),
        "split_rule": "SHA-256(source_id) modulo 3; zero is holdout, otherwise train",
        "strong_gate": {
            "best_target_token_rank": 1,
            "minimum_target_attention_mass": STRONG_ATTENTION,
            "minimum_target_pair_write_energy_fraction": STRONG_ENERGY,
        },
        "sampling_config": {
            "line_candidate_train_per_family_domain": args.line_candidate_train,
            "line_candidate_holdout_per_family_domain": args.line_candidate_holdout,
            "copy_candidate_train_per_domain": args.copy_candidate_train,
            "copy_candidate_holdout_per_domain": args.copy_candidate_holdout,
            "line_final_train_per_family_domain": args.line_final_train,
            "line_final_holdout_per_family_domain": args.line_final_holdout,
            "copy_final_train_per_domain": args.copy_final_train,
            "copy_final_holdout_per_domain": args.copy_final_holdout,
        },
        "L1H3": line_manifest,
        "L17H0": copy_manifest,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "analysis": "natural_headvis_prompt_corpus_validation",
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
        "smoke_report": str(smoke),
        "smoke_sha256": _sha256(smoke),
        "headvis_assets": {
            "sequences_sha256": _sha256(sequences_path),
            "pt_l1_umap_sha256": _sha256(l1_umap_path),
            "pt_l17_umap_sha256": _sha256(l17_umap_path),
        },
        "pt_same_index_audit": _pt_same_index_audit(sequences, l1_umap, l17_umap),
        "candidate_construction": {
            "line_candidate_events": len(line_chosen),
            "copy_candidate_events": len(copy_chosen),
            "line_prompts": len(line_prompts),
            "copy_prompts": len(copy_prompts),
            "selection_is_label_blind_to_head_activity": True,
            "sampling_config": manifest["sampling_config"],
        },
        "L1H3": {
            "layer": 1,
            "head": 3,
            "candidate_summary": _summary(line_records),
            "selected_summary": _summary(selected_line),
            "selected_event_count": len(selected_line),
            "selected_prompt_count": line_manifest["prompt_count"],
            "extraction_audit": line_audit,
            "elapsed_seconds": line_seconds,
            "examples": list(selected_line[:15]),
        },
        "L17H0": {
            "layer": 17,
            "head": 0,
            "candidate_summary": _summary(copy_records),
            "selected_summary": _summary(selected_copy),
            "selected_event_count": len(selected_copy),
            "selected_prompt_count": copy_manifest["prompt_count"],
            "extraction_audit": copy_audit,
            "elapsed_seconds": copy_seconds,
            "examples": list(selected_copy[:15]),
        },
    }
    return report, manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-report", type=Path, required=True)
    parser.add_argument("--headvis-sequences", type=Path, required=True)
    parser.add_argument("--pt-l1-umap", type=Path, required=True)
    parser.add_argument("--pt-l17-umap", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--line-candidate-train", type=int, default=10)
    parser.add_argument("--line-candidate-holdout", type=int, default=5)
    parser.add_argument("--copy-candidate-train", type=int, default=12)
    parser.add_argument("--copy-candidate-holdout", type=int, default=6)
    parser.add_argument("--line-final-train", type=int, default=3)
    parser.add_argument("--line-final-holdout", type=int, default=2)
    parser.add_argument("--copy-final-train", type=int, default=6)
    parser.add_argument("--copy-final-holdout", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    numeric = (
        args.line_candidate_train,
        args.line_candidate_holdout,
        args.copy_candidate_train,
        args.copy_candidate_holdout,
        args.line_final_train,
        args.line_final_holdout,
        args.copy_final_train,
        args.copy_final_holdout,
        args.batch_size,
        args.threads,
    )
    if min(numeric) <= 0:
        raise SystemExit("Numeric controls must be positive.")
    torch.set_num_threads(args.threads)
    report, manifest = run(args)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "natural_headvis_prompt_corpus.json"
    manifest_path = output / "natural_headvis_prompt_manifest.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(report_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
