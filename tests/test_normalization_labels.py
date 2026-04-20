from __future__ import annotations

import unittest

from postprocess_vis.normalization_labels import format_plot_label


class TestNormalizationLabels(unittest.TestCase):
    def test_rms_label_uses_std_notation(self) -> None:
        label = format_plot_label(r"$|\mathbf{u}|$", value_normalization="global_rms")
        self.assertEqual(label, r"$|\mathbf{u}| / \mathrm{std}\left(|\mathbf{u}|\right)$")

    def test_max_label_uses_max_notation(self) -> None:
        label = format_plot_label(r"$Q$", value_normalization="global_max")
        self.assertEqual(label, r"$Q / \max\left(Q\right)$")

    def test_star_label_wraps_normalized_expression(self) -> None:
        label = format_plot_label(
            r"$|\boldsymbol{\omega}|$",
            value_normalization="global_rms",
            extra_normalization="vorticity",
        )
        self.assertEqual(
            label,
            r"$\left(|\boldsymbol{\omega}| / \mathrm{std}\left(|\boldsymbol{\omega}|\right)\right)^*$",
        )

    def test_plain_star_label_preserves_absolute_value_bars(self) -> None:
        label = format_plot_label(r"$|\mathbf{u}|$", extra_normalization="velocity")
        self.assertEqual(label, r"$|\mathbf{u}^*|$")


if __name__ == "__main__":
    unittest.main()
