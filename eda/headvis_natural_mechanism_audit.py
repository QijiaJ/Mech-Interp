"""Discover and confirm natural mechanisms for IT L1H3 and L17H0.

This is label-aware exploratory analysis, not an attention-unit fit.  It uses
source-disjoint natural HeadVis corpus documents, enumerates rule opportunities
without reading head activity, and then measures exact attention and OV writes.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
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

from eda.natural_headvis_prompt_corpus import (
    Candidate,
    DOMAINS,
    STOPWORDS,
    STRONG_ATTENTION,
    STRONG_ENERGY,
    TokenizedSource,
    _overlap_positions,
    _source_from_tokens,
    _stable_digest,
    strong_event,
)
from unifying_attention.experiments.unlearned_projector_real import (
    load_registered_model_and_tokenizer,
    resolve_registered_snapshot,
)
from unifying_attention.gemma import extract_layer_qk
from unifying_attention.unlearned_projector_data import (
    compute_message_basis,
    head_output_block,
)
from unifying_attention.unlearned_projector_gate import require_valid_smoke_report


DTYPE = torch.float64
SCHEMA_VERSION = 1
MAX_EVENTS_PER_SOURCE_FAMILY = 5


@dataclass(frozen=True)
class ScoredEvent:
    record: Mapping[str, Any]
    message_contribution: torch.Tensor


def source_split(source_id: str) -> str:
    value = int(_stable_digest("natural-mechanism-split", source_id)[:8], 16)
    return "confirmation" if value % 2 else "discovery"


def _clean_piece(token: str) -> str:
    return "".join(character for character in token.strip().lower() if character.isalnum())


def _event(
    source: TokenizedSource,
    family: str,
    query: int,
    targets: Iterable[int],
    annotation: str,
) -> Candidate:
    return Candidate(
        source.source_id,
        source.domain,
        source_split(source.source_id),
        family,
        query,
        tuple((int(position), int(position) + 1) for position in sorted(set(targets))),
        annotation,
    )


def _cap(rows: Sequence[Candidate], maximum_per_family: int) -> tuple[Candidate, ...]:
    grouped: dict[str, list[Candidate]] = defaultdict(list)
    for row in rows:
        grouped[row.family].append(row)
    kept = []
    for family, values in sorted(grouped.items()):
        values.sort(
            key=lambda row: _stable_digest(
                "event-cap", row.source_id, family, row.query, row.annotation
            )
        )
        kept.extend(values[:maximum_per_family])
    return tuple(kept)


def l1_rule_events(
    source: TokenizedSource,
    maximum_per_family: int = MAX_EVENTS_PER_SOURCE_FAMILY,
) -> tuple[Candidate, ...]:
    """Enumerate local-composition opportunities without looking at L1H3."""
    rows: list[Candidate] = []
    occupied: set[tuple[int, str]] = set()

    for match in re.finditer(r"[^\W\d_]{6,}", source.text, re.UNICODE):
        positions = _overlap_positions(source.offsets, match.start(), match.end())
        if len(positions) >= 2:
            query = positions[-1]
            rows.append(_event(source, "alphabetic_word", query, positions[:-1], match.group()))
            occupied.add((query, "alphabetic_word"))

    for match in re.finditer(r"(?<!\w)\d[\d,.:/-]*\d(?!\w)", source.text):
        positions = _overlap_positions(source.offsets, match.start(), match.end())
        if len(positions) < 2:
            continue
        digits = "".join(character for character in match.group() if character.isdigit())
        family = "four_digit_year" if match.group().isdigit() and len(digits) == 4 else "other_number"
        rows.append(_event(source, family, positions[-1], positions[:-1], match.group()))

    for match in re.finditer(r"\b[A-Za-z_][A-Za-z0-9_]{5,}\b", source.text):
        surface = match.group()
        if not ("_" in surface or any(character.isdigit() for character in surface) or re.search(r"[a-z][A-Z]", surface)):
            continue
        positions = _overlap_positions(source.offsets, match.start(), match.end())
        if len(positions) >= 2 and (positions[-1], "alphabetic_word") not in occupied:
            rows.append(_event(source, "identifier", positions[-1], positions[:-1], surface))

    for match in re.finditer(r"(?:https?://|/)[^\s]{6,}", source.text):
        positions = _overlap_positions(source.offsets, match.start(), match.end())
        for left, query in zip(positions, positions[1:]):
            rows.append(_event(source, "path_or_url_piece", query, (left,), match.group()[:80]))

    for match in re.finditer(r"[^\w\s]{2,}", source.text, re.UNICODE):
        positions = _overlap_positions(source.offsets, match.start(), match.end())
        if len(positions) >= 2:
            rows.append(_event(source, "punctuation_run", positions[-1], positions[:-1], match.group()))

    newlines = [index for index, token in enumerate(source.tokens) if "\n" in token]
    for previous, query in zip(newlines, newlines[1:]):
        if query > previous + 1:
            rows.append(_event(source, "previous_newline", query, (previous,), "newline"))
    for offset, query in enumerate(newlines[1:], start=1):
        prior_newlines = [position for position in newlines[:offset] if query > position + 1]
        if prior_newlines:
            rows.append(
                _event(
                    source,
                    "newline_boundary",
                    query,
                    prior_newlines[-8:],
                    "any of the eight most recent earlier newlines",
                )
            )

    content = [
        index
        for index, token in enumerate(source.tokens)
        if len(_clean_piece(token)) >= 3
    ]
    for previous, query in zip(content, content[1:]):
        left_end = source.offsets[previous][1]
        right_start = source.offsets[query][0]
        if query > previous and any(character.isspace() for character in source.text[left_end:right_start]):
            rows.append(
                _event(source, "whitespace_previous_control", query, (previous,), "adjacent words")
            )
    return _cap(rows, maximum_per_family)


def l17_rule_events(
    source: TokenizedSource,
    maximum_per_family: int = MAX_EVENTS_PER_SOURCE_FAMILY,
) -> tuple[Candidate, ...]:
    """Enumerate retrieval opportunities without looking at L17H0."""
    rows: list[Candidate] = []
    prior: dict[int, list[int]] = defaultdict(list)
    prior_normalized: dict[str, list[int]] = defaultdict(list)
    ids = source.input_ids
    for query, current in enumerate(ids[:-1]):
        prior[current].append(query)
        current_piece = _clean_piece(source.tokens[query])
        if current_piece:
            prior_normalized[current_piece].append(query)
        next_id = ids[query + 1]
        next_piece = _clean_piece(source.tokens[query + 1])
        if len(next_piece) >= 3 and next_piece not in STOPWORDS:
            copies = [position for position in prior.get(next_id, ()) if query - position >= 5]
            induction = [
                position + 1
                for position in prior.get(current, ())[:-1]
                if position + 1 < query and ids[position + 1] == next_id
            ]
            normalized_copies = [
                position
                for position in prior_normalized.get(next_piece, ())
                if query - position >= 5
            ]
            normalized_induction = [
                position + 1
                for position in prior_normalized.get(current_piece, ())[:-1]
                if position + 1 < query
                and _clean_piece(source.tokens[position + 1]) == next_piece
            ]
            if induction:
                rows.append(
                    _event(source, "strict_induction", query, induction[-8:], next_piece)
                )
            elif normalized_induction:
                rows.append(
                    _event(
                        source,
                        "tokenization_tolerant_induction",
                        query,
                        normalized_induction[-8:],
                        next_piece,
                    )
                )
            elif copies:
                rows.append(
                    _event(source, "loose_next_token_copy", query, copies[-8:], next_piece)
                )
            elif normalized_copies:
                rows.append(
                    _event(
                        source,
                        "tokenization_tolerant_copy",
                        query,
                        normalized_copies[-8:],
                        next_piece,
                    )
                )
        same = [position for position in prior.get(current, ())[:-1] if query - position >= 5]
        if len(current_piece) >= 3 and current_piece not in STOPWORDS and same:
            rows.append(_event(source, "same_token_retrieval", query, same[-8:], current_piece))
    return _cap(rows, maximum_per_family)


def _choose_sources(
    sources: Sequence[TokenizedSource], *, per_domain_split: int
) -> tuple[TokenizedSource, ...]:
    grouped: dict[tuple[str, str], list[TokenizedSource]] = defaultdict(list)
    for source in sources:
        grouped[(source.domain, source_split(source.source_id))].append(source)
    chosen = []
    for domain in DOMAINS:
        for split in ("discovery", "confirmation"):
            values = grouped[(domain, split)]
            values.sort(key=lambda source: _stable_digest("mechanism-source", source.source_id))
            if len(values) < per_domain_split:
                raise RuntimeError(f"Insufficient natural sources for {domain}/{split}.")
            chosen.extend(values[:per_domain_split])
    return tuple(chosen)


def _pad_sources(
    sources: Sequence[TokenizedSource], pad_token_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = max(len(source.input_ids) for source in sources)
    ids = torch.full((len(sources), maximum), int(pad_token_id), dtype=torch.long)
    lengths = torch.tensor([len(source.input_ids) for source in sources], dtype=torch.long)
    for index, source in enumerate(sources):
        ids[index, : len(source.input_ids)] = torch.tensor(source.input_ids, dtype=torch.long)
    return ids, lengths


def _score_event(
    data: Any,
    source: TokenizedSource,
    event: Candidate,
    *,
    batch_index: int,
    head: int,
    message_basis: torch.Tensor,
) -> ScoredEvent:
    query = event.query
    targets = sorted({position for start, end in event.targets for position in range(start, end)})
    causal = data.source_mask[batch_index, query].nonzero().flatten()
    attention = data.attention[batch_index, head, query, causal].to(DTYPE)
    messages = data.messages[batch_index, head, causal].to(DTYPE) @ message_basis
    energy = attention.square() * messages.square().sum(1)
    target_mask = torch.tensor([int(position) in targets for position in causal], dtype=torch.bool)
    target_attention = float(attention[target_mask].sum())
    target_energy = float(energy[target_mask].sum() / energy.sum().clamp_min(1e-30))
    best_target = float(attention[target_mask].max())
    rank = 1 + int((attention > best_target).sum())
    contribution = (attention[target_mask, None] * messages[target_mask]).sum(0)
    bos_local = (causal == 0).nonzero().flatten()
    bos_attention = float(attention[bos_local].sum()) if len(bos_local) else 0.0
    bos_energy = float(energy[bos_local].sum() / energy.sum().clamp_min(1e-30)) if len(bos_local) else 0.0
    record = {
        "source_id": source.source_id,
        "domain": source.domain,
        "split": event.split,
        "family": event.family,
        "query_position": query,
        "minimum_target_distance": query - max(targets),
        "query_token": source.tokens[query],
        "target_tokens": [source.tokens[position] for position in targets],
        "target_positions": targets,
        "rule_annotation": event.annotation,
        "next_token": source.tokens[query + 1] if query + 1 < len(source.tokens) else None,
        "target_attention_mass": target_attention,
        "target_pair_write_energy_fraction": target_energy,
        "best_target_token_rank": rank,
        "bos_attention_mass": bos_attention,
        "bos_pair_write_energy_fraction": bos_energy,
        "excerpt": "".join(source.tokens[max(0, query - 18) : min(len(source.tokens), query + 10)]),
    }
    record["strong_behavior_gate"] = strong_event(record)
    return ScoredEvent(record=record, message_contribution=contribution.cpu())


def _route_category(source: TokenizedSource, query: int, key: int) -> str:
    ids = source.input_ids
    if key == 0:
        return "bos"
    if key == query:
        return "self"
    if query + 1 < len(ids) and ids[key] == ids[query + 1]:
        if key > 0 and ids[key - 1] == ids[query]:
            return "strict_induction"
        return "next_token_copy"
    key_piece = _clean_piece(source.tokens[key])
    query_piece = _clean_piece(source.tokens[query])
    next_piece = _clean_piece(source.tokens[query + 1]) if query + 1 < len(ids) else ""
    if key_piece and key_piece == next_piece:
        if key > 0 and _clean_piece(source.tokens[key - 1]) == query_piece:
            return "tokenization_tolerant_induction"
        return "tokenization_tolerant_copy"
    if ids[key] == ids[query]:
        return "same_token"
    if key_piece and key_piece == query_piece:
        return "tokenization_tolerant_same_token"
    if "\n" in source.tokens[key] and "\n" in source.tokens[query]:
        return "newline_to_newline"
    if key < query:
        start = source.offsets[key][0]
        end = source.offsets[query][1]
        span = source.text[start:end].strip()
        if span and not any(character.isspace() for character in span):
            if span.isalpha():
                return "within_alpha_surface"
            if any(character.isdigit() for character in span) and all(
                character.isdigit() or character in ",.:/-" for character in span
            ):
                return "within_numeric_surface"
            if any(character.isalnum() for character in span):
                return "within_mixed_surface"
            return "within_punctuation_surface"
    return "local_other" if query - key <= 2 else "long_other"


def _inventory_routes(
    data: Any,
    source: TokenizedSource,
    *,
    batch_index: int,
    head: int,
    message_basis: torch.Tensor,
) -> list[Mapping[str, Any]]:
    rows = []
    length = len(source.input_ids)
    for query in range(1, length - 1):
        causal = data.source_mask[batch_index, query].nonzero().flatten()
        attention = data.attention[batch_index, head, query, causal].to(DTYPE)
        messages = data.messages[batch_index, head, causal].to(DTYPE) @ message_basis
        energy = attention.square() * messages.square().sum(1)
        local = int(attention.argmax())
        key = int(causal[local])
        record = {
            "source_id": source.source_id,
            "domain": source.domain,
            "split": source_split(source.source_id),
            "category": _route_category(source, query, key),
            "query_position": query,
            "key_position": key,
            "distance": query - key,
            "query_token": source.tokens[query],
            "key_token": source.tokens[key],
            "next_token": source.tokens[query + 1],
            "attention": float(attention[local]),
            "pair_write_energy_fraction": float(energy[local] / energy.sum().clamp_min(1e-30)),
            "excerpt": "".join(source.tokens[max(0, query - 14) : min(length, query + 8)]),
        }
        record["strong_route"] = bool(
            record["attention"] >= STRONG_ATTENTION
            and record["pair_write_energy_fraction"] >= STRONG_ENERGY
        )
        rows.append(record)
    return rows


def _event_summary(rows: Sequence[ScoredEvent]) -> Mapping[str, Any]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    domain_grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row.record["family"], row.record["split"])].append(row.record)
        domain_grouped[
            (row.record["family"], row.record["domain"], row.record["split"])
        ].append(row.record)

    def summarize(values: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        strong = [value for value in values if value["strong_behavior_gate"]]
        return {
            "opportunities": len(values),
            "strong_events": len(strong),
            "strong_rate": len(strong) / len(values),
            "mean_target_attention": sum(value["target_attention_mass"] for value in values) / len(values),
            "mean_target_write_energy": sum(value["target_pair_write_energy_fraction"] for value in values) / len(values),
            "top1_rate": sum(value["best_target_token_rank"] == 1 for value in values) / len(values),
            "unique_sources_with_strong_events": len({value["source_id"] for value in strong}),
        }

    return {
        "by_family_split": {
            "|".join(key): summarize(values)
            for key, values in sorted(grouped.items())
        },
        "by_family_domain_split": {
            "|".join(key): summarize(values)
            for key, values in sorted(domain_grouped.items())
        },
        "opportunities": len(rows),
        "strong_events": sum(bool(row.record["strong_behavior_gate"]) for row in rows),
    }


def _route_summary(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["strong_route"]:
            grouped[(row["category"], row["split"])].append(row)
    return {
        "strong_routes": sum(bool(row["strong_route"]) for row in rows),
        "all_queries": len(rows),
        "by_category_split": {
            "|".join(key): {
                "routes": len(values),
                "unique_sources": len({value["source_id"] for value in values}),
                "domains": sorted({value["domain"] for value in values}),
                "mean_attention": sum(value["attention"] for value in values) / len(values),
                "mean_write_energy": sum(value["pair_write_energy_fraction"] for value in values) / len(values),
                "mean_distance": sum(value["distance"] for value in values) / len(values),
            }
            for key, values in sorted(grouped.items())
        },
    }


def _top_basis(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    covariance = rows.T @ rows
    values, vectors = torch.linalg.eigh(covariance)
    order = torch.argsort(values, descending=True, stable=True)
    values = values.index_select(0, order).clamp_min(0)
    total = values.sum()
    rank = 0 if float(total) <= 1e-30 else int(torch.searchsorted(values.cumsum(0), 0.9 * total)) + 1
    return vectors.index_select(1, order[:rank]), values, rank


def _message_geometry(rows: Sequence[ScoredEvent]) -> Mapping[str, Any]:
    grouped: dict[tuple[str, str], list[torch.Tensor]] = defaultdict(list)
    for row in rows:
        if row.record["strong_behavior_gate"]:
            grouped[(row.record["family"], row.record["split"])].append(row.message_contribution)
    bases: dict[tuple[str, str], torch.Tensor] = {}
    records: dict[str, Any] = {}
    for key, values in sorted(grouped.items()):
        matrix = torch.stack(values).to(DTYPE)
        basis, _spectrum, rank = _top_basis(matrix)
        bases[key] = basis
        records["|".join(key)] = {"events": len(values), "effective_rank_90": rank}
    transfer = {}
    families = sorted({key[0] for key in grouped})
    for family in families:
        discovery = grouped.get((family, "discovery"), [])
        confirmation = grouped.get((family, "confirmation"), [])
        frame = bases.get((family, "discovery"))
        if not discovery or not confirmation or frame is None or not frame.shape[1]:
            continue
        matrix = torch.stack(confirmation).to(DTYPE)
        fraction = (matrix @ frame).square().sum() / matrix.square().sum().clamp_min(1e-30)
        transfer[family] = float(fraction)
    overlap = {}
    for left_index, left in enumerate(families):
        for right in families[left_index + 1 :]:
            a = bases.get((left, "discovery"))
            b = bases.get((right, "discovery"))
            if a is None or b is None or not a.shape[1] or not b.shape[1]:
                continue
            singular = torch.linalg.svdvals(a.T @ b).clamp(0, 1)
            overlap[f"{left}|{right}"] = float(singular.square().sum() / min(a.shape[1], b.shape[1]))
    return {
        "by_family_split": records,
        "discovery_subspace_confirmation_energy_capture": transfer,
        "discovery_pairwise_normalized_overlap": overlap,
    }


def _examples(
    rows: Sequence[Mapping[str, Any]], *, key: str, per_group: int = 12
) -> Mapping[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    result = {}
    for group, values in sorted(grouped.items()):
        values.sort(
            key=lambda row: (
                -(row.get("attention", row.get("target_attention_mass", 0.0))
                  * row.get("pair_write_energy_fraction", row.get("target_pair_write_energy_fraction", 0.0))),
                row["source_id"],
            )
        )
        seen = set()
        chosen = []
        for value in values:
            if value["source_id"] in seen:
                continue
            seen.add(value["source_id"])
            chosen.append(value)
            if len(chosen) == per_group:
                break
        result[group] = chosen
    return result


def _run_head(
    model: Any,
    tokenizer: Any,
    sources: Sequence[TokenizedSource],
    events: Mapping[str, Sequence[Candidate]],
    *,
    layer: int,
    head: int,
    batch_size: int,
    chunk_size: int,
    inventory_routes: bool = True,
    include_strong_manifest: bool = False,
) -> Mapping[str, Any]:
    message_basis = compute_message_basis(
        head_output_block(model, layer_idx=layer, head=head), expected_rank=256
    ).basis
    scored: list[ScoredEvent] = []
    routes: list[Mapping[str, Any]] = []
    audits = []
    started = time.perf_counter()
    for start in range(0, len(sources), chunk_size):
        chunk = sources[start : start + chunk_size]
        ids, lengths = _pad_sources(chunk, tokenizer.pad_token_id)
        data, audit = extract_layer_qk(
            model,
            input_ids=ids,
            prompt_lengths=lengths,
            layer_idx=layer,
            batch_size=batch_size,
            recompute_ov_float64=True,
        )
        audits.append(audit)
        for batch_index, source in enumerate(chunk):
            for event in events[source.source_id]:
                scored.append(
                    _score_event(
                        data,
                        source,
                        event,
                        batch_index=batch_index,
                        head=head,
                        message_basis=message_basis,
                    )
                )
            if inventory_routes:
                routes.extend(
                    _inventory_routes(
                        data,
                        source,
                        batch_index=batch_index,
                        head=head,
                        message_basis=message_basis,
                    )
                )
        del data
    audit = {
        "chunks": len(audits),
        "maximum_attention_error": max(row["attention_max_abs_error"] for row in audits),
        "maximum_head_output_error": max(row["head_output_max_abs_error"] for row in audits),
        "post_rope": all(row["post_rope"] for row in audits),
        "symmetric_scaling": all(row["symmetric_scaling"] for row in audits),
        "ov_recomputed_float64": all(row["ov_recomputed_float64"] for row in audits),
    }
    strong_records = [row.record for row in scored if row.record["strong_behavior_gate"]]
    strong_routes = [row for row in routes if row["strong_route"]]
    result = {
        "layer": layer,
        "head": head,
        "source_count": len(sources),
        "elapsed_seconds": time.perf_counter() - started,
        "extraction_audit": audit,
        "rule_opportunity_summary": _event_summary(scored),
        "top_route_inventory": _route_summary(routes),
        "message_geometry": _message_geometry(scored),
        "strong_rule_examples": _examples(strong_records, key="family"),
        "strong_top_route_examples": _examples(strong_routes, key="category"),
    }
    if include_strong_manifest:
        result["strong_event_manifest"] = [
            {key: value for key, value in record.items() if key != "excerpt"}
            for record in strong_records
        ]
    return result


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    smoke = Path(args.smoke_report).resolve()
    require_valid_smoke_report(smoke)
    sequences_path = Path(args.headvis_sequences).resolve()
    sequences = json.loads(sequences_path.read_text())
    artifacts = resolve_registered_snapshot(allow_download=False)
    model, tokenizer = load_registered_model_and_tokenizer(artifacts, allow_download=False)
    try:
        all_sources = [
            _source_from_tokens(tokenizer, str(source_id), exported)
            for source_id, exported in sequences.items()
        ]
        sources = _choose_sources(all_sources, per_domain_split=args.sources_per_domain_split)
        l1_events = {source.source_id: l1_rule_events(source) for source in sources}
        l17_events = {source.source_id: l17_rule_events(source) for source in sources}
        l1 = _run_head(
            model,
            tokenizer,
            sources,
            l1_events,
            layer=1,
            head=3,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
        )
        l17 = _run_head(
            model,
            tokenizer,
            sources,
            l17_events,
            layer=17,
            head=0,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
        )
    finally:
        del model
        del tokenizer
    source_rows = [
        {
            "source_id": source.source_id,
            "domain": source.domain,
            "split": source_split(source.source_id),
            "text_sha256": hashlib.sha256(source.text.encode()).hexdigest(),
        }
        for source in sources
    ]
    discovery = {row["source_id"] for row in source_rows if row["split"] == "discovery"}
    confirmation = {row["source_id"] for row in source_rows if row["split"] == "confirmation"}
    if discovery & confirmation:
        raise RuntimeError("Natural mechanism source split leaked.")
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis": "headvis_natural_mechanism_audit",
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
        "headvis_sequences_sha256": hashlib.sha256(sequences_path.read_bytes()).hexdigest(),
        "sampling": {
            "source_count": len(sources),
            "sources_per_domain_split": args.sources_per_domain_split,
            "domains": list(DOMAINS),
            "maximum_events_per_source_family": MAX_EVENTS_PER_SOURCE_FAMILY,
            "split": "SHA256 natural-mechanism-split(source_id) parity",
            "source_disjoint": True,
            "candidate_enumeration_reads_head_activity": False,
            "strong_gate": {
                "target_rank": 1,
                "minimum_attention": STRONG_ATTENTION,
                "minimum_pair_write_energy": STRONG_ENERGY,
            },
        },
        "sources": source_rows,
        "L1H3": l1,
        "L17H0": l17,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-report", type=Path, required=True)
    parser.add_argument("--headvis-sequences", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources-per-domain-split", type=int, default=20)
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
