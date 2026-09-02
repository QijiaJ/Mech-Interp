from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "spectral" / "code" / "spectral_cluster.py"
SPEC = importlib.util.spec_from_file_location("spectral_cluster", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
spectral = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = spectral
SPEC.loader.exec_module(spectral)


def bank(q: torch.Tensor, keys: torch.Tensor, messages: torch.Tensor, weights: torch.Tensor):
    count = len(q)
    return spectral.EventBank(
        target="synthetic",
        split="train",
        q=q,
        keys_attention_center=keys,
        keys_uniform_center=keys,
        messages=messages,
        weights=weights,
        source_ids=tuple(map(str, range(count))),
        query_positions=torch.arange(count),
        semantic_labels=torch.arange(count) % 2,
        subtype_labels=torch.arange(count) % 2,
        families=tuple("ab"[i % 2] for i in range(count)),
        domains=tuple("test" for _ in range(count)),
        layout=torch.zeros(count, 3, dtype=torch.float64),
        retained_attention=torch.ones(count, dtype=torch.float64),
        retained_attention_squared=torch.ones(count, dtype=torch.float64),
    )


class SpectralKernelTest(unittest.TestCase):
    def test_block_gram_matches_direct_and_is_psd(self) -> None:
        generator = torch.Generator().manual_seed(3)
        q = torch.randn(7, 4, generator=generator)
        keys = torch.randn(7, 3, 4, generator=generator)
        messages = torch.randn(7, 3, 5, generator=generator)
        weights = torch.rand(7, 3, generator=generator)
        weights = weights / weights.sum(1, keepdim=True)
        data = bank(q, keys, messages, weights)
        actual = spectral._tensor_gram(
            data, data, center="attention", block=3, symmetric=True
        ).double()
        expected = torch.empty_like(actual)
        for left in range(7):
            for right in range(7):
                value = 0.0
                for j in range(3):
                    for k in range(3):
                        value += float(
                            weights[left, j]
                            * weights[right, k]
                            * (keys[left, j] @ keys[right, k]).square()
                            * (messages[left, j] @ messages[right, k]).square()
                        )
                expected[left, right] = value
        self.assertTrue(torch.allclose(actual, expected, rtol=2e-5, atol=2e-5))
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(actual).min()), -2e-4)

    def test_normalised_kernel_has_unit_diagonal(self) -> None:
        generator = torch.Generator().manual_seed(5)
        q = torch.randn(6, 4, generator=generator)
        keys = torch.randn(6, 2, 4, generator=generator)
        messages = torch.randn(6, 2, 3, generator=generator)
        weights = torch.full((6, 2), 0.5)
        data = bank(q, keys, messages, weights)
        tensor = spectral._tensor_gram(
            data, data, center="attention", block=4, symmetric=True
        )
        tensor_self = spectral._tensor_self(data, center="attention")
        q_cross = q @ q.T
        q_self = q.square().sum(1)
        for mode in ("add", "product"):
            kernel, _, _ = spectral._normalised_kernel(
                q_cross,
                tensor,
                q_self,
                q_self,
                tensor_self,
                tensor_self,
                mode=mode,
                gamma=1.0,
            )
            self.assertTrue(torch.allclose(kernel.diag(), torch.ones(6, dtype=torch.float64)))
            self.assertGreaterEqual(float(kernel.min()), 0.0)
            self.assertLessEqual(float(kernel.max()), 1.0 + 1e-12)


if __name__ == "__main__":
    unittest.main()
