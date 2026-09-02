"""Small geometry helpers used by the label-aware corpus diagnostics."""
from __future__ import annotations

from typing import Any, Mapping

import torch


DTYPE = torch.float64


def _canonical_columns(frame: torch.Tensor) -> torch.Tensor:
    if frame.numel() == 0:
        return frame
    columns = []
    for column in frame.T:
        pivot = int(column.abs().argmax())
        columns.append(column if float(column[pivot]) >= 0 else -column)
    return torch.stack(columns, dim=1)


def top_basis(rows: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the top uncentered covariance basis and descending spectrum."""
    if rows.ndim != 2 or rank < 0 or rank > rows.shape[1]:
        raise ValueError("Rows/rank are invalid for a top covariance basis.")
    covariance = rows.T @ rows
    values, vectors = torch.linalg.eigh(covariance)
    order = torch.argsort(values, descending=True, stable=True)
    spectrum = values.index_select(0, order).clamp_min(0)
    basis = vectors.index_select(1, order[:rank])
    return _canonical_columns(basis), spectrum


def effective_rank_90(spectrum: torch.Tensor) -> int:
    total = spectrum.clamp_min(0).sum()
    if float(total) <= 1e-30:
        return 0
    return int(torch.searchsorted(spectrum.cumsum(0), 0.9 * total).item()) + 1


def normalized_overlap(left: torch.Tensor, right: torch.Tensor) -> Mapping[str, Any]:
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError("Subspace frames must have one common ambient width.")
    if not left.shape[1] or not right.shape[1]:
        return {
            "normalized_overlap": 0.0,
            "principal_cosine_squared": [],
            "minimum_cosine_squared": 0.0,
            "median_cosine_squared": 0.0,
        }
    singular = torch.linalg.svdvals(left.T @ right).clamp(0, 1)
    cos2 = singular.square()
    return {
        "normalized_overlap": float(cos2.sum() / min(left.shape[1], right.shape[1])),
        "principal_cosine_squared": [float(value) for value in cos2.tolist()],
        "minimum_cosine_squared": float(cos2.min()),
        "median_cosine_squared": float(cos2.median()),
    }


def _standardize(train: torch.Tensor, other: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train.mean(0)
    scale = (train - mean).square().mean(0).sqrt()
    positive = scale[scale > 0]
    minimum = 1e-8 * float(positive.median()) if len(positive) else torch.finfo(train.dtype).eps
    scale = scale.clamp_min(minimum)
    return (train - mean) / scale, (other - mean) / scale


def _confusion(predicted: torch.Tensor, truth: torch.Tensor, count: int) -> list[list[int]]:
    return [
        [int(((truth == actual) & (predicted == guess)).sum()) for guess in range(count)]
        for actual in range(count)
    ]


def _accuracy_record(
    predicted: torch.Tensor,
    truth: torch.Tensor,
    prompts: torch.Tensor,
    *,
    class_count: int,
) -> Mapping[str, Any]:
    correct = (predicted == truth).to(DTYPE)
    prompt_ids = torch.unique(prompts, sorted=True)
    prompt_accuracy = torch.stack([correct[prompts == prompt].mean() for prompt in prompt_ids])
    recall = [float(correct[truth == family].mean()) for family in range(class_count)]
    return {
        "accuracy": float(correct.mean()),
        "macro_recall": sum(recall) / len(recall),
        "prompt_balanced_accuracy": float(prompt_accuracy.mean()),
        "recall_by_family": recall,
        "confusion_truth_by_prediction": _confusion(predicted, truth, class_count),
        "observations": len(truth),
        "prompts": len(prompt_ids),
    }


def nearest_centroid(
    train: torch.Tensor,
    train_labels: torch.Tensor,
    train_prompts: torch.Tensor,
    holdout: torch.Tensor,
    holdout_labels: torch.Tensor,
    holdout_prompts: torch.Tensor,
    *,
    class_count: int,
) -> Mapping[str, Any]:
    """Fit TRAIN-only PCA90 and labeled centroids, then evaluate both splits."""
    standardized_train, standardized_holdout = _standardize(train, holdout)
    centered = standardized_train - standardized_train.mean(0)
    _basis_full, spectrum = top_basis(centered, min(centered.shape))
    rank = max(1, effective_rank_90(spectrum))
    basis, _ = top_basis(centered, rank)
    train_z = standardized_train @ basis
    holdout_z = standardized_holdout @ basis
    centroids = torch.stack([train_z[train_labels == family].mean(0) for family in range(class_count)])
    train_prediction = torch.cdist(train_z, centroids).square().argmin(1)
    holdout_prediction = torch.cdist(holdout_z, centroids).square().argmin(1)
    overall = train_z.mean(0)
    within = torch.stack(
        [
            (train_z[train_labels == family] - centroids[family]).square().sum(1).mean()
            for family in range(class_count)
        ]
    ).mean()
    between = (centroids - overall).square().sum(1).mean()
    return {
        "rank": rank,
        "train_accuracy": _accuracy_record(
            train_prediction, train_labels, train_prompts, class_count=class_count
        )["accuracy"],
        "holdout_accuracy": _accuracy_record(
            holdout_prediction, holdout_labels, holdout_prompts, class_count=class_count
        )["accuracy"],
        "train": _accuracy_record(
            train_prediction, train_labels, train_prompts, class_count=class_count
        ),
        "holdout": _accuracy_record(
            holdout_prediction, holdout_labels, holdout_prompts, class_count=class_count
        ),
        "train_between_to_within": float(between / within.clamp_min(1e-30)),
    }
