"""Plot data for Algorithm 2's exact event-level initialization geometry."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

import algorithm2 as alg
from algorithm1 import SourceRows, _configure_imports, _extract_head, _fit_common_span, _kmeans
from investigate_label_algorithm_2 import _coordinates, _fit_coordinate_model, _truth


DTYPE = torch.float64


def _project_3d(train_x: torch.Tensor, holdout_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train = train_x.to(DTYPE)
    holdout = holdout_x.to(DTYPE)
    mean = train.mean(0, keepdim=True)
    covariance = (train - mean).T @ (train - mean)
    values, vectors = torch.linalg.eigh((covariance + covariance.T) / 2)
    order = torch.argsort(values, descending=True, stable=True)
    basis = vectors[:, order[:3]]
    explained = values[order[:3]] / values.clamp_min(0).sum()
    return (train - mean) @ basis, (holdout - mean) @ basis, explained


def _nearest_centroid_metrics(x: torch.Tensor, truth: torch.Tensor) -> Mapping[str, float]:
    centers = torch.stack([x[truth == unit].mean(0) for unit in (0, 1)])
    predicted = torch.cdist(x, centers).argmin(1)
    recalls = [float((predicted[truth == unit] == unit).to(DTYPE).mean()) for unit in (0, 1)]
    within = torch.sqrt(torch.stack([
        (x[truth == unit] - centers[unit]).square().sum(1).mean()
        for unit in (0, 1)
    ]))
    return {
        "balanced_accuracy": sum(recalls) / 2,
        "recall_class_0": recalls[0],
        "recall_class_1": recalls[1],
        "centroid_distance": float((centers[0] - centers[1]).norm()),
        "mean_within_class_rms_radius": float(within.mean()),
        "centroid_distance_over_mean_radius": float((centers[0] - centers[1]).norm() / within.mean()),
    }


def _round_rows(x: torch.Tensor) -> list[list[float]]:
    return [[round(float(value), 4) for value in row] for row in x]


def _head(raw: Sequence[SourceRows], target: str) -> Mapping[str, Any]:
    _v0, _complement, _c0 = _fit_common_span(raw)
    prepared = alg._prepare_sources(raw)
    train = [source for source in prepared if source.base.split == "train"]
    holdout = [source for source in prepared if source.base.split == "holdout"]
    model = _fit_coordinate_model(train)
    train_x, train_lengths = _coordinates(train, model)
    holdout_x, holdout_lengths = _coordinates(holdout, model)
    train_kmeans = _kmeans(train_x, 2, 0)
    kmeans_centers = torch.stack([train_x[train_kmeans == unit].mean(0) for unit in (0, 1)])
    holdout_kmeans = torch.cdist(holdout_x, kmeans_centers).argmin(1)
    train_3d, holdout_3d, explained = _project_3d(train_x, holdout_x)
    train_truth = torch.cat([_truth(source, target) for source in train])
    holdout_truth = torch.cat([_truth(source, target) for source in holdout])
    # Project the frozen K-means centers using the same TRAIN mean and basis by
    # recovering them as group means in the already projected coordinates.
    centers_3d = torch.stack([train_3d[train_kmeans == unit].mean(0) for unit in (0, 1)])
    return {
        "target": target,
        "truth_names": ["word-like", "newline"] if target == "L1H1" else ["strict induction", "tokenization-tolerant induction"],
        "explained_variance_fraction": [float(value) for value in explained],
        "train": {"xyz": _round_rows(train_3d), "truth": train_truth.tolist(), "kmeans": train_kmeans.tolist()},
        "holdout": {"xyz": _round_rows(holdout_3d), "truth": holdout_truth.tolist(), "kmeans": holdout_kmeans.tolist()},
        "kmeans_centers_xyz": _round_rows(centers_3d),
        "full_6d_truth_centroid_diagnostic": {
            "train": _nearest_centroid_metrics(train_x.to(DTYPE), train_truth),
            "holdout_using_holdout_centroids_descriptive_only": _nearest_centroid_metrics(holdout_x.to(DTYPE), holdout_truth),
        },
        "event_counts": {"train": train_lengths, "holdout": holdout_lengths},
    }


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
    model, tokenizer = load_registered_model_and_tokenizer(artifacts, allow_download=False)
    try:
        l1, _ = _extract_head(model, tokenizer, corpus, target="L1H1", layer=1, head=1, chunk_size=args.chunk_size, batch_size=args.batch_size)
        l5, _ = _extract_head(model, tokenizer, corpus, target="L5H2", layer=5, head=2, chunk_size=args.chunk_size, batch_size=args.batch_size)
    finally:
        del model
        del tokenizer
    return {
        "schema_version": 1,
        "analysis": "algorithm2_initialization_geometry_3d",
        "development_only": True,
        "locked_test_accessed": False,
        "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        "projection": "top three PCA directions of the exact six-dimensional standardized TRAIN initialization coordinates; holdout projected without refitting",
        "L1H1": _head(l1, "L1H1"),
        "L5H2": _head(l5, "L5H2"),
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
    args.output.write_text(json.dumps(result, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
