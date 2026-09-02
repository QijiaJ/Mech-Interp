"""Ground-truth EDA for the frozen natural IT L1H1/L5H2 corpus.

This script measures train/holdout relation and message geometry, important
versus background queries, shared/background versus behavior-specific message
structure, and pair-label versus event-label diagnostics. It fits no latent
attention-unit estimator and performs no model selection.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import html
import json
import math
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

from eda.geometry import (
    effective_rank_90,
    nearest_centroid,
    normalized_overlap,
    top_basis,
)
from eda.headvis_natural_mechanism_audit import _pad_sources
from eda.natural_headvis_prompt_corpus import TokenizedSource
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
BACKGROUND_QUERIES_PER_SOURCE = 8
L1_SEMANTIC = {
    "alphabetic_word": 0,
    "four_digit_year": 0,
    "identifier": 0,
    "other_number": 0,
    "newline_boundary": 1,
}
L1_SEMANTIC_NAMES = ("surface_composition", "newline_structure")
L1_FINE_NAMES = tuple(sorted(L1_SEMANTIC))
L1_FINE = {name: index for index, name in enumerate(L1_FINE_NAMES)}
L5_SUBTYPE_NAMES = ("strict_induction", "tokenization_tolerant_induction")
L5_SUBTYPE = {name: index for index, name in enumerate(L5_SUBTYPE_NAMES)}
MODALITIES = (
    "Q",
    "target_K",
    "Q_plus_target_K",
    "target_source_message",
    "target_contribution",
    "full_event_write",
    "nontarget_write",
)
PAIR_MODALITIES = ("QK_pair", "V_source_message")
COLORS = ("#2d6cdf", "#e4572e", "#2ca02c", "#9467bd", "#8c564b", "#17becf")


@dataclass
class ExtractedHead:
    important: Mapping[str, torch.Tensor]
    background: Mapping[str, torch.Tensor]
    metadata: Mapping[str, Any]
    pair_audit: Mapping[str, Any]
    extraction_audit: Mapping[str, Any]


class WeightedCentroid:
    """Streaming standardized nearest-centroid fit and holdout scorer."""

    def __init__(self, width: int, class_count: int) -> None:
        self.width = width
        self.class_count = class_count
        self.weight = 0.0
        self.sum = torch.zeros(width, dtype=DTYPE)
        self.square_sum = torch.zeros(width, dtype=DTYPE)
        self.class_weight = torch.zeros(class_count, dtype=DTYPE)
        self.class_sum = torch.zeros(class_count, width, dtype=DTYPE)
        self.observations = 0
        self._fit: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self.correct_weight = 0.0
        self.holdout_weight = 0.0
        self.holdout_observations = 0
        self.confusion = torch.zeros(class_count, class_count, dtype=DTYPE)

    def update_train(
        self, rows: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor
    ) -> None:
        if not len(rows):
            return
        rows = rows.to(DTYPE)
        weights = weights.to(DTYPE)
        self.weight += float(weights.sum())
        self.sum += (weights[:, None] * rows).sum(0)
        self.square_sum += (weights[:, None] * rows.square()).sum(0)
        self.observations += len(rows)
        for label in range(self.class_count):
            mask = labels == label
            if mask.any():
                local_weight = weights[mask]
                self.class_weight[label] += local_weight.sum()
                self.class_sum[label] += (local_weight[:, None] * rows[mask]).sum(0)

    def finish(self) -> None:
        if self.weight <= 0 or (self.class_weight <= 0).any():
            raise RuntimeError("Every pair-centroid class must have positive training weight.")
        mean = self.sum / self.weight
        variance = (self.square_sum / self.weight - mean.square()).clamp_min(0)
        scale = variance.sqrt()
        positive = scale[scale > 0]
        floor = 1e-8 * float(positive.median()) if len(positive) else 1e-12
        scale = scale.clamp_min(floor)
        centroids = (self.class_sum / self.class_weight[:, None] - mean) / scale
        self._fit = mean, scale, centroids

    def update_holdout(
        self, rows: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor
    ) -> None:
        if not len(rows):
            return
        if self._fit is None:
            raise RuntimeError("Pair centroid must be finalized before holdout scoring.")
        mean, scale, centroids = self._fit
        z = (rows.to(DTYPE) - mean) / scale
        prediction = torch.cdist(z, centroids).square().argmin(1)
        weights = weights.to(DTYPE)
        correct = prediction == labels
        self.correct_weight += float(weights[correct].sum())
        self.holdout_weight += float(weights.sum())
        self.holdout_observations += len(rows)
        for truth in range(self.class_count):
            for guess in range(self.class_count):
                mask = (labels == truth) & (prediction == guess)
                self.confusion[truth, guess] += weights[mask].sum()

    def record(self) -> Mapping[str, Any]:
        recall = []
        for label in range(self.class_count):
            total = self.confusion[label].sum()
            recall.append(float(self.confusion[label, label] / total) if total > 0 else 0.0)
        return {
            "train_observations": self.observations,
            "train_weight": self.weight,
            "holdout_observations": self.holdout_observations,
            "holdout_weight": self.holdout_weight,
            "holdout_accuracy": self.correct_weight / max(self.holdout_weight, 1e-30),
            "holdout_macro_recall": sum(recall) / len(recall),
            "holdout_recall_by_class": recall,
            "holdout_weighted_confusion": self.confusion.tolist(),
        }


class PairAudit:
    def __init__(self, class_count: int) -> None:
        widths = {"QK_pair": 512, "V_source_message": 256}
        self.models = {
            scope: {
                name: WeightedCentroid(widths[name], class_count)
                for name in PAIR_MODALITIES
            }
            for scope in ("all_pairs", "energy_weighted_pairs", "target_pairs")
        }
        self.event_count = defaultdict(int)
        self.pair_count = defaultdict(int)
        self.target_pair_count = defaultdict(int)
        self.target_energy_fraction_sum = defaultdict(float)
        self.target_attention_sum = defaultdict(float)

    @staticmethod
    def _features(q: torch.Tensor, keys: torch.Tensor, messages: torch.Tensor) -> Mapping[str, torch.Tensor]:
        repeated = q[None, :].expand(len(keys), -1)
        return {
            "QK_pair": torch.cat((repeated, keys), dim=1),
            "V_source_message": messages,
        }

    def update(
        self,
        *,
        split: str,
        label: int,
        q: torch.Tensor,
        keys: torch.Tensor,
        messages: torch.Tensor,
        attention: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> None:
        labels = torch.full((len(keys),), label, dtype=torch.long)
        energy = attention.square() * messages.square().sum(1)
        features = self._features(q, keys, messages)
        self.event_count[split] += 1
        self.pair_count[split] += len(keys)
        self.target_pair_count[split] += int(target_mask.sum())
        self.target_energy_fraction_sum[split] += float(
            energy[target_mask].sum() / energy.sum().clamp_min(1e-30)
        )
        self.target_attention_sum[split] += float(attention[target_mask].sum())
        for name, rows in features.items():
            calls = (
                ("all_pairs", rows, labels, torch.ones(len(rows), dtype=DTYPE)),
                ("energy_weighted_pairs", rows, labels, energy),
                (
                    "target_pairs",
                    rows[target_mask],
                    labels[target_mask],
                    torch.ones(int(target_mask.sum()), dtype=DTYPE),
                ),
            )
            for scope, local_rows, local_labels, weights in calls:
                model = self.models[scope][name]
                if split == "train":
                    model.update_train(local_rows, local_labels, weights)
                else:
                    model.update_holdout(local_rows, local_labels, weights)

    def finish_train(self) -> None:
        for models in self.models.values():
            for model in models.values():
                model.finish()

    def record(self) -> Mapping[str, Any]:
        return {
            "semantics": (
                "Every causal pair inherits its query-event label in all-pair scopes; "
                "target_pairs contains only prespecified behavioral targets."
            ),
            "counts": {
                split: {
                    "events": self.event_count[split],
                    "pairs": self.pair_count[split],
                    "target_pairs": self.target_pair_count[split],
                    "target_pair_count_fraction": self.target_pair_count[split]
                    / max(self.pair_count[split], 1),
                    "macro_mean_target_attention": self.target_attention_sum[split]
                    / max(self.event_count[split], 1),
                    "macro_mean_target_pair_write_energy_fraction": self.target_energy_fraction_sum[split]
                    / max(self.event_count[split], 1),
                }
                for split in ("train", "holdout")
            },
            "centroid_transfer": {
                scope: {name: model.record() for name, model in models.items()}
                for scope, models in self.models.items()
            },
        }


def _stable_key(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()


def _tokenize_source(tokenizer: Any, row: Mapping[str, Any]) -> TokenizedSource:
    encoded = tokenizer(row["text"], add_special_tokens=True, return_offsets_mapping=True)
    ids = tuple(int(value) for value in encoded["input_ids"])
    tokens = tuple(tokenizer.decode([value], skip_special_tokens=False) for value in ids)
    if "".join(tokens) != "<bos>" + row["text"]:
        raise RuntimeError(f"Frozen source {row['source_id']} failed tokenizer round trip.")
    return TokenizedSource(
        source_id=str(row["source_id"]),
        text=str(row["text"]),
        input_ids=ids,
        offsets=tuple((int(a), int(b)) for a, b in encoded["offset_mapping"]),
        tokens=tokens,
        domain=str(row["domain"]),
    )


def _stack(rows: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(rows).to(DTYPE) if rows else torch.empty((0, 0), dtype=DTYPE)


def _extract_head(
    model: Any,
    tokenizer: Any,
    corpus: Mapping[str, Any],
    *,
    target: str,
    layer: int,
    head: int,
    batch_size: int,
    chunk_size: int,
) -> ExtractedHead:
    source_rows = {str(row["source_id"]): row for row in corpus["sources"]}
    sources = {key: _tokenize_source(tokenizer, row) for key, row in source_rows.items()}
    events_by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for event in corpus[target]["events"]:
        events_by_source[str(event["source_id"])].append(event)
    basis = compute_message_basis(
        head_output_block(model, layer_idx=layer, head=head), expected_rank=256
    ).basis
    important: dict[str, list[torch.Tensor]] = defaultdict(list)
    background: dict[str, list[torch.Tensor]] = defaultdict(list)
    important_meta: dict[str, list[Any]] = defaultdict(list)
    background_meta: dict[str, list[Any]] = defaultdict(list)
    class_count = 2
    pair_audit = PairAudit(class_count)
    audits = []
    source_index = {source_id: index for index, source_id in enumerate(sorted(sources, key=int))}

    def process(split: str) -> None:
        selected = [
            sources[source_id]
            for source_id, row in source_rows.items()
            if row["split"] == split
        ]
        selected.sort(key=lambda source: int(source.source_id))
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
            audits.append(audit)
            for batch_index, source in enumerate(chunk):
                event_rows = sorted(
                    events_by_source.get(source.source_id, ()),
                    key=lambda row: int(row["query_position"]),
                )
                annotated = {int(row["query_position"]) for row in event_rows}
                for event in event_rows:
                    if event["split"] != split:
                        raise RuntimeError("Event and source split disagree.")
                    query = int(event["query_position"])
                    targets = {int(value) for value in event["target_positions"]}
                    if source.tokens[query] != event["query_token"]:
                        raise RuntimeError("Frozen query token no longer matches tokenizer output.")
                    causal = data.source_mask[batch_index, query].nonzero().flatten()
                    attention = data.attention[batch_index, head, query, causal].to(DTYPE)
                    keys = data.keys[batch_index, head, causal].to(DTYPE)
                    messages = data.messages[batch_index, head, causal].to(DTYPE) @ basis
                    q = data.queries[batch_index, head, query].to(DTYPE)
                    target_mask = torch.tensor(
                        [int(position) in targets for position in causal], dtype=torch.bool
                    )
                    target_mass = attention[target_mask].sum().clamp_min(1e-30)
                    target_k = (attention[target_mask, None] * keys[target_mask]).sum(0) / target_mass
                    attended_k = (attention[:, None] * keys).sum(0)
                    target_source = (
                        attention[target_mask, None] * messages[target_mask]
                    ).sum(0) / target_mass
                    target_write = (
                        attention[target_mask, None] * messages[target_mask]
                    ).sum(0)
                    full_write = (attention[:, None] * messages).sum(0)
                    nontarget_write = full_write - target_write
                    values = {
                        "Q": q,
                        "target_K": target_k,
                        "attended_K": attended_k,
                        "Q_plus_target_K": torch.cat((q, target_k)),
                        "target_source_message": target_source,
                        "target_contribution": target_write,
                        "full_event_write": full_write,
                        "nontarget_write": nontarget_write,
                        "layout_control": torch.tensor(
                            [
                                query / max(1, len(source.input_ids) - 1),
                                len(source.input_ids) / 512.0,
                                (query - max(targets)) / max(1, len(source.input_ids) - 1),
                                len(targets) / 8.0,
                            ],
                            dtype=DTYPE,
                        ),
                    }
                    for name, value in values.items():
                        important[name].append(value.cpu())
                    family = str(event["family"])
                    semantic = L1_SEMANTIC[family] if target == "L1H1" else 0
                    subtype = L1_FINE[family] if target == "L1H1" else L5_SUBTYPE[family]
                    important_meta["semantic_label"].append(semantic)
                    important_meta["subtype_label"].append(subtype)
                    important_meta["source"].append(source_index[source.source_id])
                    important_meta["domain"].append(source.domain)
                    important_meta["family"].append(family)
                    important_meta["split"].append(split)
                    important_meta["query_position"].append(query)
                    important_meta["target_mass"].append(float(target_mass))
                    energy = attention.square() * messages.square().sum(1)
                    important_meta["target_energy_fraction"].append(
                        float(energy[target_mask].sum() / energy.sum().clamp_min(1e-30))
                    )
                    important_meta["bos_top"].append(int(causal[int(attention.argmax())]) == 0)
                    pair_label = semantic if target == "L1H1" else subtype
                    pair_audit.update(
                        split=split,
                        label=pair_label,
                        q=q,
                        keys=keys,
                        messages=messages,
                        attention=attention,
                        target_mask=target_mask,
                    )

                candidates = [
                    query
                    for query in range(1, len(source.input_ids) - 1)
                    if query not in annotated and bool(data.valid_queries[batch_index, query])
                ]
                candidates.sort(key=lambda query: _stable_key(target, source.source_id, query))
                for query in candidates[:BACKGROUND_QUERIES_PER_SOURCE]:
                    causal = data.source_mask[batch_index, query].nonzero().flatten()
                    attention = data.attention[batch_index, head, query, causal].to(DTYPE)
                    keys = data.keys[batch_index, head, causal].to(DTYPE)
                    messages = data.messages[batch_index, head, causal].to(DTYPE) @ basis
                    q = data.queries[batch_index, head, query].to(DTYPE)
                    attended_k = (attention[:, None] * keys).sum(0)
                    full_write = (attention[:, None] * messages).sum(0)
                    for name, value in {
                        "Q": q,
                        "attended_K": attended_k,
                        "Q_plus_attended_K": torch.cat((q, attended_k)),
                        "full_event_write": full_write,
                    }.items():
                        background[name].append(value.cpu())
                    background_meta["source"].append(source_index[source.source_id])
                    background_meta["domain"].append(source.domain)
                    background_meta["split"].append(split)
                    background_meta["query_position"].append(query)
                    background_meta["bos_top"].append(int(causal[int(attention.argmax())]) == 0)
            del data
        if split == "train":
            pair_audit.finish_train()

    process("train")
    process("holdout")

    def finalize(values: Mapping[str, list[Any]], meta: Mapping[str, list[Any]]) -> Mapping[str, torch.Tensor]:
        result: dict[str, Any] = {name: _stack(rows) for name, rows in values.items()}
        for name in ("semantic_label", "subtype_label", "source", "query_position", "bos_top"):
            if name in meta:
                result[name] = torch.tensor(meta[name], dtype=torch.long)
        result["split"] = tuple(meta["split"])
        result["domain"] = tuple(meta["domain"])
        if "family" in meta:
            result["family"] = tuple(meta["family"])
        for name in ("target_mass", "target_energy_fraction"):
            if name in meta:
                result[name] = torch.tensor(meta[name], dtype=DTYPE)
        return result

    return ExtractedHead(
        important=finalize(important, important_meta),
        background=finalize(background, background_meta),
        metadata={
            "target": target,
            "layer": layer,
            "head": head,
            "background_queries_per_source": BACKGROUND_QUERIES_PER_SOURCE,
        },
        pair_audit=pair_audit.record(),
        extraction_audit={
            "chunks": len(audits),
            "maximum_attention_error": max(row["attention_max_abs_error"] for row in audits),
            "maximum_head_output_error": max(row["head_output_max_abs_error"] for row in audits),
            "post_rope": all(row["post_rope"] for row in audits),
            "symmetric_scaling": all(row["symmetric_scaling"] for row in audits),
            "ov_recomputed_float64": all(row["ov_recomputed_float64"] for row in audits),
        },
    )


def _split_mask(table: Mapping[str, Any], split: str) -> torch.Tensor:
    return torch.tensor([value == split for value in table["split"]], dtype=torch.bool)


def _cluster_record(
    table: Mapping[str, Any], labels: torch.Tensor, *, class_count: int
) -> Mapping[str, Any]:
    train = _split_mask(table, "train")
    holdout = _split_mask(table, "holdout")
    records = {}
    for name in MODALITIES:
        records[name] = nearest_centroid(
            table[name][train],
            labels[train],
            table["source"][train],
            table[name][holdout],
            labels[holdout],
            table["source"][holdout],
            class_count=class_count,
        )
    return records


def _nuisance_residual_cluster_record(
    table: Mapping[str, Any], labels: torch.Tensor, *, class_count: int
) -> Mapping[str, Any]:
    """Remove train-fitted linear layout/domain effects before classification."""
    train = _split_mask(table, "train")
    holdout = _split_mask(table, "holdout")
    domain_names = tuple(sorted(set(table["domain"])))

    def design(mask: torch.Tensor) -> torch.Tensor:
        indices = mask.nonzero().flatten().tolist()
        domain = torch.tensor(
            [
                [float(table["domain"][index] == name) for name in domain_names]
                for index in indices
            ],
            dtype=DTYPE,
        )
        return torch.cat(
            (
                torch.ones((len(indices), 1), dtype=DTYPE),
                table["layout_control"][mask],
                domain,
            ),
            dim=1,
        )

    train_design = design(train)
    holdout_design = design(holdout)
    records = {}
    for name in MODALITIES:
        coefficient = torch.linalg.lstsq(train_design, table[name][train]).solution
        train_residual = table[name][train] - train_design @ coefficient
        holdout_residual = table[name][holdout] - holdout_design @ coefficient
        records[name] = nearest_centroid(
            train_residual,
            labels[train],
            table["source"][train],
            holdout_residual,
            labels[holdout],
            table["source"][holdout],
            class_count=class_count,
        )
    return {
        "nuisance_design": (
            "intercept + normalized query position + sequence length + normalized "
            "nearest-target distance + target count + five domain indicators"
        ),
        "centroid_transfer_on_residuals": records,
    }


def _active_background_clusters(head: ExtractedHead) -> Mapping[str, Any]:
    important = head.important
    background = head.background
    train_i = _split_mask(important, "train")
    hold_i = _split_mask(important, "holdout")
    train_b = _split_mask(background, "train")
    hold_b = _split_mask(background, "holdout")
    pairs = {
        "Q": (important["Q"], background["Q"]),
        "attended_K": (important["attended_K"], background["attended_K"]),
        "QK": (
            torch.cat((important["Q"], important["attended_K"]), dim=1),
            background["Q_plus_attended_K"],
        ),
        "full_event_write": (important["full_event_write"], background["full_event_write"]),
    }
    records = {}
    for name, (active, quiet) in pairs.items():
        train_rows = torch.cat((active[train_i], quiet[train_b]))
        train_labels = torch.cat(
            (torch.zeros(int(train_i.sum()), dtype=torch.long), torch.ones(int(train_b.sum()), dtype=torch.long))
        )
        train_sources = torch.cat((important["source"][train_i], background["source"][train_b]))
        hold_rows = torch.cat((active[hold_i], quiet[hold_b]))
        hold_labels = torch.cat(
            (torch.zeros(int(hold_i.sum()), dtype=torch.long), torch.ones(int(hold_b.sum()), dtype=torch.long))
        )
        hold_sources = torch.cat((important["source"][hold_i], background["source"][hold_b]))
        records[name] = nearest_centroid(
            train_rows,
            train_labels,
            train_sources,
            hold_rows,
            hold_labels,
            hold_sources,
            class_count=2,
        )
    return {
        "labels": ("strong_prespecified_event", "sampled_unannotated_query"),
        "sample_counts": {
            "train_important": int(train_i.sum()),
            "holdout_important": int(hold_i.sum()),
            "train_background": int(train_b.sum()),
            "holdout_background": int(hold_b.sum()),
        },
        "bos_top_rate": {
            "train_important": float(important["bos_top"][train_i].to(DTYPE).mean()),
            "holdout_important": float(important["bos_top"][hold_i].to(DTYPE).mean()),
            "train_background": float(background["bos_top"][train_b].to(DTYPE).mean()),
            "holdout_background": float(background["bos_top"][hold_b].to(DTYPE).mean()),
        },
        "centroid_transfer": records,
    }


def _subspace_transfer(
    rows: torch.Tensor,
    labels: torch.Tensor,
    splits: Sequence[str],
    names: Sequence[str],
) -> Mapping[str, Any]:
    train_mask = torch.tensor([value == "train" for value in splits], dtype=torch.bool)
    holdout_mask = ~train_mask
    records = {}
    for label, name in enumerate(names):
        train = rows[train_mask & (labels == label)]
        holdout = rows[holdout_mask & (labels == label)]
        train_basis_full, train_spectrum = top_basis(train, min(train.shape))
        hold_basis_full, hold_spectrum = top_basis(holdout, min(holdout.shape))
        train_rank = max(1, effective_rank_90(train_spectrum))
        hold_rank = max(1, effective_rank_90(hold_spectrum))
        train_basis = train_basis_full[:, :train_rank]
        hold_basis = hold_basis_full[:, :hold_rank]
        records[name] = {
            "train_events": len(train),
            "holdout_events": len(holdout),
            "train_r90": train_rank,
            "holdout_r90": hold_rank,
            **normalized_overlap(train_basis, hold_basis),
            "holdout_energy_captured_by_train_r90": float(
                (holdout @ train_basis).square().sum() / holdout.square().sum().clamp_min(1e-30)
            ),
            "train_energy_captured_by_holdout_r90": float(
                (train @ hold_basis).square().sum() / train.square().sum().clamp_min(1e-30)
            ),
        }
    return records


def _message_transfer(head: ExtractedHead, *, target: str) -> Mapping[str, Any]:
    table = head.important
    if target == "L1H1":
        label_sets = {"semantic": (table["semantic_label"], L1_SEMANTIC_NAMES)}
    else:
        label_sets = {
            "semantic_pooled": (torch.zeros(len(table["Q"]), dtype=torch.long), ("induction",)),
            "address_subtype": (table["subtype_label"], L5_SUBTYPE_NAMES),
        }
    return {
        scope: {
            name: _subspace_transfer(table[name], labels, table["split"], names)
            for name in (
                "target_source_message",
                "target_contribution",
                "full_event_write",
                "nontarget_write",
            )
        }
        for scope, (labels, names) in label_sets.items()
    }


def _capture(rows: torch.Tensor, basis: torch.Tensor) -> float:
    return float((rows @ basis).square().sum() / rows.square().sum().clamp_min(1e-30))


def _common_specific(head: ExtractedHead, *, target: str) -> Mapping[str, Any]:
    table = head.important
    background_queries = head.background
    train = _split_mask(table, "train")
    holdout = _split_mask(table, "holdout")
    bg_train = _split_mask(background_queries, "train")
    bg_hold = _split_mask(background_queries, "holdout")
    labels = table["semantic_label"]
    names = L1_SEMANTIC_NAMES if target == "L1H1" else ("induction",)
    class_count = len(names)

    train_background = table["nontarget_write"][train]
    hold_background = table["nontarget_write"][holdout]
    background_basis_full, background_spectrum = top_basis(
        train_background, min(train_background.shape)
    )
    shared_rank = max(1, effective_rank_90(background_spectrum))
    shared = background_basis_full[:, :shared_rank]
    random_train = background_queries["full_event_write"][bg_train]
    random_hold = background_queries["full_event_write"][bg_hold]
    random_basis_full, random_spectrum = top_basis(random_train, min(random_train.shape))
    random_rank = max(1, effective_rank_90(random_spectrum))
    random_basis = random_basis_full[:, :random_rank]

    target_train = table["target_contribution"][train]
    target_hold = table["target_contribution"][holdout]
    train_labels = labels[train]
    hold_labels = labels[holdout]
    train_residual = target_train - (target_train @ shared) @ shared.T
    hold_residual = target_hold - (target_hold @ shared) @ shared.T
    unique = []
    unique_full = []
    unique_ranks = []
    for label in range(class_count):
        local = train_residual[train_labels == label]
        basis_full, spectrum = top_basis(local, min(local.shape))
        rank = max(1, effective_rank_90(spectrum))
        unique_full.append(basis_full)
        unique.append(basis_full[:, :rank])
        unique_ranks.append(rank)

    shared_only_error = float(hold_residual.square().sum() / target_hold.square().sum().clamp_min(1e-30))
    reconstructed = (target_hold @ shared) @ shared.T
    for label, basis in enumerate(unique):
        mask = hold_labels == label
        reconstructed[mask] += (hold_residual[mask] @ basis) @ basis.T
    matched_error = float(
        (target_hold - reconstructed).square().sum() / target_hold.square().sum().clamp_min(1e-30)
    )
    swapped_error = None
    if class_count == 2:
        wrong = (target_hold @ shared) @ shared.T
        for label in range(class_count):
            other = 1 - label
            # Preserve the receiving family's accessible rank. Otherwise the
            # 104-versus-6 L1 rank asymmetry would make an ordinary swap
            # uninformative about family specificity.
            basis = unique_full[other][:, : unique_ranks[label]]
            mask = hold_labels == label
            wrong[mask] += (hold_residual[mask] @ basis) @ basis.T
        swapped_error = float(
            (target_hold - wrong).square().sum() / target_hold.square().sum().clamp_min(1e-30)
        )

    by_family = {}
    for label, name in enumerate(names):
        mask = hold_labels == label
        basis = unique[label]
        by_family[name] = {
            "unique_r90_after_background_projection": unique_ranks[label],
            "target_energy_captured_by_background_V0": _capture(target_hold[mask], shared),
            "residual_target_energy_captured_by_matching_unique": _capture(
                hold_residual[mask], basis
            ),
        }
    return {
        "definition": (
            "V0 is train r90 PCA of the non-target contribution inside strong events; "
            "Va is train r90 PCA of each target contribution after projecting out V0."
        ),
        "background_V0_rank": shared_rank,
        "random_unannotated_query_rank": random_rank,
        "background_V0_train_holdout_capture": _capture(hold_background, shared),
        "random_background_train_holdout_capture": _capture(random_hold, random_basis),
        "within_event_background_vs_random_query_overlap": normalized_overlap(shared, random_basis),
        "random_query_capture_by_within_event_background_V0": _capture(random_hold, shared),
        "target_shared_only_normalized_error": shared_only_error,
        "target_shared_plus_matching_unique_normalized_error": matched_error,
        "target_wrong_unique_normalized_error": swapped_error,
        "wrong_minus_matched_error": None if swapped_error is None else swapped_error - matched_error,
        "by_family": by_family,
    }


def _important_background_margin(
    head: ExtractedHead, *, target: str
) -> Mapping[str, Any]:
    table = head.important
    background = head.background
    labels = table["semantic_label"] if target == "L1H1" else table["subtype_label"]
    names = L1_SEMANTIC_NAMES if target == "L1H1" else L5_SUBTYPE_NAMES
    train = _split_mask(table, "train")
    hold = _split_mask(table, "holdout")
    bg_hold = _split_mask(background, "holdout")
    records = {}
    mapping = {
        "Q": (table["Q"], background["Q"]),
        "attended_K": (table["attended_K"], background["attended_K"]),
        "QK": (
            torch.cat((table["Q"], table["attended_K"]), dim=1),
            background["Q_plus_attended_K"],
        ),
        "full_event_write": (table["full_event_write"], background["full_event_write"]),
    }
    for name, (important_rows, background_rows) in mapping.items():
        train_rows = important_rows[train]
        mean = train_rows.mean(0)
        scale = (train_rows - mean).square().mean(0).sqrt()
        positive = scale[scale > 0]
        floor = 1e-8 * float(positive.median()) if len(positive) else 1e-12
        scale = scale.clamp_min(floor)
        train_z = (train_rows - mean) / scale
        centered = train_z - train_z.mean(0)
        basis_full, spectrum = top_basis(centered, min(centered.shape))
        rank = max(1, effective_rank_90(spectrum))
        basis = basis_full[:, :rank]
        train_low = train_z @ basis
        centroids = torch.stack(
            [train_low[labels[train] == label].mean(0) for label in range(len(names))]
        )
        within = torch.stack(
            [
                (train_low[labels[train] == label] - centroids[label]).square().sum(1).mean()
                for label in range(len(names))
            ]
        ).mean()

        def distances(rows: torch.Tensor) -> tuple[float, float]:
            low = ((rows - mean) / scale) @ basis
            dist = torch.cdist(low, centroids).square()
            nearest = dist.min(1).values
            if len(names) == 2:
                ordered = dist.sort(1).values
                margin = (ordered[:, 1] - ordered[:, 0]) / (ordered[:, 1] + ordered[:, 0]).clamp_min(1e-30)
                median_margin = float(margin.median())
            else:
                median_margin = 0.0
            return float((nearest / within.clamp_min(1e-30)).median()), median_margin

        important_distance, important_margin = distances(important_rows[hold])
        background_distance, background_margin = distances(background_rows[bg_hold])
        records[name] = {
            "pca90_rank": rank,
            "holdout_important_median_nearest_distance_over_train_within": important_distance,
            "holdout_background_median_nearest_distance_over_train_within": background_distance,
            "holdout_important_median_two_class_margin": important_margin,
            "holdout_background_median_two_class_margin": background_margin,
        }
    return records


def _pca_xy(train: torch.Tensor, holdout: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    mean = train.mean(0)
    centered = train - mean
    basis, spectrum = top_basis(centered, 2)
    explained = float(spectrum[:2].sum() / spectrum.sum().clamp_min(1e-30))
    return centered @ basis, (holdout - mean) @ basis, explained


def _write_pca_svg(
    path: Path,
    head: ExtractedHead,
    *,
    target: str,
    labels: torch.Tensor,
    names: Sequence[str],
) -> None:
    table = head.important
    train = _split_mask(table, "train")
    holdout = _split_mask(table, "holdout")
    panels = (
        ("Q", table["Q"]),
        ("target K", table["target_K"]),
        ("Q + target K", table["Q_plus_target_K"]),
        ("target source message", table["target_source_message"]),
        ("target contribution", table["target_contribution"]),
        ("full event write", table["full_event_write"]),
    )
    width, height = 1000, 680
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="28" font-family="sans-serif" font-size="18" font-weight="bold">{html.escape(target)}: train-fitted PCA of strong natural events</text>',
        '<text x="20" y="48" font-family="sans-serif" font-size="11" fill="#444">circles: train; crosses: source-disjoint holdout. PCA is fit separately per panel on train.</text>',
    ]
    for index, (title, rows) in enumerate(panels):
        row_index, column = divmod(index, 3)
        x0, y0 = 20 + column * 325, 70 + row_index * 280
        train_xy, hold_xy, explained = _pca_xy(rows[train], rows[holdout])
        combined = torch.cat((train_xy, hold_xy))
        low = combined.quantile(.01, dim=0)
        high = combined.quantile(.99, dim=0)
        span = (high - low).clamp_min(1e-12)

        def location(value: torch.Tensor) -> tuple[float, float]:
            scaled = ((value - low) / span).clamp(0, 1)
            return x0 + 25 + 260 * float(scaled[0]), y0 + 225 - 190 * float(scaled[1])

        parts.extend(
            (
                f'<rect x="{x0}" y="{y0}" width="305" height="245" fill="#fafafa" stroke="#bbb"/>',
                f'<text x="{x0+8}" y="{y0+18}" font-family="sans-serif" font-size="12" font-weight="bold">{html.escape(title)}</text>',
                f'<text x="{x0+8}" y="{y0+34}" font-family="sans-serif" font-size="10" fill="#555">PC1+2={explained:.3f}</text>',
            )
        )
        train_indices = torch.arange(len(train_xy))[:: max(1, math.ceil(len(train_xy) / 1400))]
        hold_indices = torch.arange(len(hold_xy))[:: max(1, math.ceil(len(hold_xy) / 900))]
        local_train_labels = labels[train]
        local_hold_labels = labels[holdout]
        for local in train_indices.tolist():
            x, y = location(train_xy[local])
            color = COLORS[int(local_train_labels[local])]
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1.4" fill="{color}" fill-opacity=".28"/>')
        for local in hold_indices.tolist():
            x, y = location(hold_xy[local])
            color = COLORS[int(local_hold_labels[local])]
            parts.append(f'<path d="M {x-2:.2f} {y-2:.2f} L {x+2:.2f} {y+2:.2f} M {x-2:.2f} {y+2:.2f} L {x+2:.2f} {y-2:.2f}" stroke="{color}" stroke-width=".8"/>')
    for index, name in enumerate(names):
        x = 30 + index * 240
        parts.append(f'<circle cx="{x}" cy="650" r="5" fill="{COLORS[index]}"/>')
        parts.append(f'<text x="{x+10}" y="654" font-family="sans-serif" font-size="11">{html.escape(name)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def _analyze_head(head: ExtractedHead, *, target: str, output_dir: Path) -> Mapping[str, Any]:
    table = head.important
    if target == "L1H1":
        primary_labels = table["semantic_label"]
        primary_names = L1_SEMANTIC_NAMES
        cluster = {
            "semantic_A2_labels": _cluster_record(table, primary_labels, class_count=2),
            "fine_rule_labels": _cluster_record(table, table["subtype_label"], class_count=len(L1_FINE_NAMES)),
        }
    else:
        primary_labels = table["subtype_label"]
        primary_names = L5_SUBTYPE_NAMES
        cluster = {
            "semantic_A1": "Only one semantic class; multiclass accuracy is undefined.",
            "address_subtype_diagnostic": _cluster_record(table, primary_labels, class_count=2),
        }
    train = _split_mask(table, "train")
    holdout = _split_mask(table, "holdout")
    cluster["layout_only_control"] = nearest_centroid(
        table["layout_control"][train],
        primary_labels[train],
        table["source"][train],
        table["layout_control"][holdout],
        primary_labels[holdout],
        table["source"][holdout],
        class_count=len(primary_names),
    )
    cluster["after_linear_layout_and_domain_residualization"] = (
        _nuisance_residual_cluster_record(
            table, primary_labels, class_count=len(primary_names)
        )
    )
    figure = output_dir / f"{target.lower()}_ground_truth_pca.svg"
    _write_pca_svg(
        figure,
        head,
        target=target,
        labels=primary_labels,
        names=primary_names,
    )
    return {
        "head": head.metadata,
        "ground_truth_interpretation": (
            "surface composition versus newline structure"
            if target == "L1H1"
            else "one induction class; exact/tolerant are address subtypes only"
        ),
        "counts": {
            "important_events": len(table["Q"]),
            "sampled_background_queries": len(head.background["Q"]),
            "important_by_split_family": {
                "|".join(key): value
                for key, value in sorted(
                    {
                        key: sum(
                            1
                            for split, family in zip(table["split"], table["family"])
                            if (split, family) == key
                        )
                        for key in set(zip(table["split"], table["family"]))
                    }.items()
                )
            },
        },
        "message_subspace_train_holdout": _message_transfer(head, target=target),
        "ground_truth_cluster_transfer": cluster,
        "important_vs_background_queries": _active_background_clusters(head),
        "semantic_centroid_proximity_of_background": _important_background_margin(head, target=target),
        "common_background_and_specific_message": _common_specific(head, target=target),
        "pair_label_vs_event_label": head.pair_audit,
        "extraction_audit": head.extraction_audit,
        "figure": figure.name,
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    require_valid_smoke_report(Path(args.smoke_report).resolve())
    corpus_path = Path(args.corpus).resolve()
    corpus = json.loads(corpus_path.read_text())
    if corpus.get("attention_unit_fit") is not False or corpus.get("model_selection_run") is not False:
        raise RuntimeError("Expected a pre-fitting natural corpus.")
    artifacts = resolve_registered_snapshot(allow_download=False)
    if corpus["model"]["revision"] != artifacts.revision:
        raise RuntimeError("Corpus/model revision mismatch.")
    model, tokenizer = load_registered_model_and_tokenizer(artifacts, allow_download=False)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        l1 = _extract_head(
            model,
            tokenizer,
            corpus,
            target="L1H1",
            layer=1,
            head=1,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
        )
        l5 = _extract_head(
            model,
            tokenizer,
            corpus,
            target="L5H2",
            layer=5,
            head=2,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
        )
    finally:
        del model
        del tokenizer
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis": "candidate_head_ground_truth_eda",
        "development_only": True,
        "ground_truth_labels_used": True,
        "attention_unit_fit": False,
        "model_selection_run": False,
        "historical_test_accessed": False,
        "locked_test_accessed": False,
        "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        "model": corpus["model"],
        "definitions": {
            "target_source_message": "attention-normalized mean of exact source messages over rule targets",
            "target_contribution": "sum over targets of exact attention times exact source message",
            "nontarget_write": "full exact event write minus target contribution",
            "full_event_write": "sum over the complete causal row of exact attention times exact source message",
            "cluster_metric": "train-only standardization, PCA90, labeled nearest centroid, source-disjoint holdout",
            "subspace_metric": "uncentered PCA r90; principal-angle overlap and cross-split energy capture",
            "pair_energy": "exact attention squared times exact source-message squared norm",
        },
        "L1H1": _analyze_head(l1, target="L1H1", output_dir=output_dir),
        "L5H2": _analyze_head(l5, target="L5H2", output_dir=output_dir),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-report", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if min(args.chunk_size, args.batch_size, args.threads) <= 0:
        raise SystemExit("All numeric controls must be positive.")
    torch.set_num_threads(args.threads)
    result = run(args)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
