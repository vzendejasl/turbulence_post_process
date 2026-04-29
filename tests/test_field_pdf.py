from __future__ import annotations

import contextlib
import io
import unittest
from unittest import mock

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

try:
    from mpi4py import MPI
except ImportError:  # pragma: no cover - environment dependent
    MPI = None

from postprocess_vis.pdfs import _configure_pdf_axes
from postprocess_vis.pdfs import _trimmed_decimal_label
from postprocess_vis.pdfs import compute_distributed_field_pdf
from postprocess_vis.pdfs import DEFAULT_FIELD_PDF_SMOOTH_SIGMA_BINS
from postprocess_vis.pdfs import DEFAULT_FIELD_PDF_SMOOTHING
from postprocess_vis.pdfs import field_pdf_output_path
from postprocess_vis.pdfs import print_field_pdf_summary
from postprocess_vis.pdfs import plot_field_pdf
from postprocess_vis.pdfs import plot_smoothed_field_pdf
from postprocess_vis.pdfs import rescale_field_pdf_for_plot
from postprocess_vis.pdfs import smooth_field_pdf_for_plot


@unittest.skipIf(MPI is None, "mpi4py is not installed in this test environment")
class TestFieldPdf(unittest.TestCase):
    def test_trimmed_decimal_label_uses_up_to_two_decimals(self) -> None:
        self.assertEqual(_trimmed_decimal_label(1.1), "1.1")
        self.assertEqual(_trimmed_decimal_label(0.95), "0.95")
        self.assertEqual(_trimmed_decimal_label(1.0), "1.0")

    def test_summary_prints_normalization_scales(self) -> None:
        pdf_result = {
            "pdf_name": "normalized_density",
            "source_field": "density",
            "source_field_min": -0.5,
            "source_field_max": 3.25,
            "source_field_mean": 0.00125,
            "source_field_std": 1.75,
            "source_field_rms": 1.8,
            "normalization": "global_std",
            "normalization_scale": 1.75,
            "normalization_offset": 0.00125,
            "measured_normalization_scale": 1.75,
            "value_range_min": -1.0,
            "value_range_max": 2.0,
            "bin_count": 16,
            "total_samples": 8,
            "pdf_integral": 1.0,
            "binning_warning": "test warning",
            "range_warning": "",
        }

        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            print_field_pdf_summary(pdf_result)
        printed = stream.getvalue()

        self.assertIn("Source field min  : -0.5", printed)
        self.assertIn("Source field max  : 3.25", printed)
        self.assertIn("Source field mean : 0.00125", printed)
        self.assertIn("Source field std  : 1.75", printed)
        self.assertIn("Source field rms  : 1.8", printed)
        self.assertIn("Normalization offset: 0.00125", printed)
        self.assertIn("Normalization scale: 1.75", printed)
        self.assertIn("Measured normalization scale: 1.75", printed)

    def test_configure_pdf_axes_caps_linear_ticks(self) -> None:
        fig, ax = plt.subplots()
        try:
            _configure_pdf_axes(
                ax,
                x_formatter=mock.Mock(),
                y_scale="linear",
            )
            self.assertIsInstance(ax.xaxis.get_major_locator(), MaxNLocator)
            self.assertIsInstance(ax.yaxis.get_major_locator(), MaxNLocator)
            self.assertEqual(ax.xaxis.get_major_locator()._nbins, 5)
            self.assertEqual(ax.yaxis.get_major_locator()._nbins, 5)
        finally:
            plt.close(fig)

    def test_plot_field_pdf_defaults_to_yt_backend(self) -> None:
        pdf_result = {
            "pdf_name": "normalized_density",
            "source_field": "density",
            "normalization": "global_std",
            "bin_edges": np.array([0.0, 1.0, 2.0], dtype=np.float64),
            "bin_centers": np.array([0.5, 1.5], dtype=np.float64),
            "counts": np.array([1, 1], dtype=np.int64),
            "pdf": np.array([0.5, 0.5], dtype=np.float64),
            "x_label": r"$\rho / \mathrm{std}(\rho)$",
            "normalization_scale": 2.0,
            "value_range_min": 0.0,
            "value_range_max": 2.0,
            "plot_title": "Normalized Density PDF",
        }

        with mock.patch("postprocess_vis.pdfs._plot_field_pdf_yt", return_value="yt-path") as yt_plot:
            with mock.patch("postprocess_vis.pdfs._plot_field_pdf_matplotlib") as mpl_plot:
                resolved = plot_field_pdf(pdf_result, "out.pdf")

        self.assertEqual(resolved, "yt-path")
        yt_plot.assert_called_once()
        self.assertEqual(yt_plot.call_args.kwargs["y_scale"], "log")
        mpl_plot.assert_not_called()

    def test_plot_field_pdf_accepts_explicit_matplotlib_backend(self) -> None:
        pdf_result = {
            "pdf_name": "normalized_density",
            "source_field": "density",
            "normalization": "global_std",
            "bin_edges": np.array([0.0, 1.0, 2.0], dtype=np.float64),
            "bin_centers": np.array([0.5, 1.5], dtype=np.float64),
            "counts": np.array([1, 1], dtype=np.int64),
            "pdf": np.array([0.5, 0.5], dtype=np.float64),
            "x_label": r"$\rho / \mathrm{std}(\rho)$",
            "normalization_scale": 2.0,
            "value_range_min": 0.0,
            "value_range_max": 2.0,
            "plot_title": "Normalized Density PDF",
        }

        with mock.patch("postprocess_vis.pdfs._plot_field_pdf_yt") as yt_plot:
            with mock.patch("postprocess_vis.pdfs._plot_field_pdf_matplotlib", return_value="mpl-path") as mpl_plot:
                resolved = plot_field_pdf(pdf_result, "out.pdf", backend="matplotlib")

        self.assertEqual(resolved, "mpl-path")
        yt_plot.assert_not_called()
        mpl_plot.assert_called_once()

    def test_plot_smoothed_field_pdf_defaults_to_yt_backend(self) -> None:
        pdf_result = {
            "pdf_name": "normalized_density",
            "source_field": "density",
            "normalization": "global_std",
            "bin_edges": np.array([0.0, 1.0, 2.0], dtype=np.float64),
            "bin_centers": np.array([0.5, 1.5], dtype=np.float64),
            "counts": np.array([1, 3], dtype=np.int64),
            "pdf": np.array([0.25, 0.75], dtype=np.float64),
            "x_label": r"$\rho / \mathrm{std}(\rho)$",
            "normalization_scale": 2.0,
            "value_range_min": 0.0,
            "value_range_max": 2.0,
            "plot_title": "Normalized Density PDF",
        }

        with mock.patch("postprocess_vis.pdfs._plot_field_pdf_yt", return_value="yt-path") as yt_plot:
            with mock.patch("postprocess_vis.pdfs._plot_field_pdf_matplotlib") as mpl_plot:
                resolved = plot_smoothed_field_pdf(pdf_result, "out.pdf")

        self.assertEqual(resolved, "yt-path")
        yt_plot.assert_called_once()
        self.assertEqual(yt_plot.call_args.kwargs["y_scale"], "log")
        mpl_plot.assert_not_called()

    def test_raw_replot_rescales_axis_and_density(self) -> None:
        pdf_result = {
            "pdf_name": "normalized_dilatation",
            "source_field": "div_u",
            "normalization": "global_std",
            "bin_edges": np.array([-2.0, 0.0, 2.0], dtype=np.float64),
            "bin_centers": np.array([-1.0, 1.0], dtype=np.float64),
            "counts": np.array([1, 1], dtype=np.int64),
            "pdf": np.array([0.25, 0.25], dtype=np.float64),
            "x_label": "normalized div_u",
            "plot_title": "Normalized Dilatation PDF",
            "measured_normalization_scale": 4.0,
            "normalization_offset": 0.0,
        }

        raw = rescale_field_pdf_for_plot(pdf_result, x_normalization="raw")

        np.testing.assert_allclose(raw["bin_edges"], np.array([-8.0, 0.0, 8.0], dtype=np.float64))
        np.testing.assert_allclose(raw["bin_centers"], np.array([-4.0, 4.0], dtype=np.float64))
        np.testing.assert_allclose(raw["pdf"], np.array([0.0625, 0.0625], dtype=np.float64))
        self.assertEqual(raw["x_label"], "div_u")

    def test_raw_replot_uses_explicit_raw_axis_label_when_available(self) -> None:
        pdf_result = {
            "pdf_name": "normalized_density",
            "source_field": "density",
            "normalization": "global_std",
            "bin_edges": np.array([0.0, 1.0, 2.0], dtype=np.float64),
            "bin_centers": np.array([0.5, 1.5], dtype=np.float64),
            "counts": np.array([1, 1], dtype=np.int64),
            "pdf": np.array([0.5, 0.5], dtype=np.float64),
            "x_label": r"$\rho / \mathrm{std}(\rho)$",
            "raw_x_label": r"$\rho$",
            "plot_title": "Normalized Density PDF",
            "normalization_scale": 2.0,
            "normalization_offset": 1.0,
        }

        raw = rescale_field_pdf_for_plot(pdf_result, x_normalization="raw")

        self.assertEqual(raw["x_label"], r"$\rho$")
        np.testing.assert_allclose(raw["bin_edges"], np.array([1.0, 3.0, 5.0], dtype=np.float64))
        np.testing.assert_allclose(raw["bin_centers"], np.array([2.0, 4.0], dtype=np.float64))

    def test_source_rms_replot_renormalizes_saved_pdf(self) -> None:
        pdf_result = {
            "pdf_name": "rms_normalized_u2",
            "source_field": "vy",
            "normalization": "global_std",
            "bin_edges": np.array([0.0, 1.0, 2.0], dtype=np.float64),
            "bin_centers": np.array([0.5, 1.5], dtype=np.float64),
            "counts": np.array([1, 1], dtype=np.int64),
            "pdf": np.array([0.5, 0.5], dtype=np.float64),
            "raw_x_label": r"$u_2$",
            "source_field_mean": 1.0,
            "source_field_std": 2.0,
            "source_field_rms": 4.0,
            "normalization_scale": 2.0,
            "normalization_offset": 1.0,
            "plot_title": "Second Velocity Component PDF",
        }

        scaled = rescale_field_pdf_for_plot(pdf_result, x_normalization="source_rms")

        np.testing.assert_allclose(scaled["bin_edges"], np.array([0.25, 0.75, 1.25], dtype=np.float64))
        np.testing.assert_allclose(scaled["bin_centers"], np.array([0.5, 1.0], dtype=np.float64))
        np.testing.assert_allclose(scaled["pdf"], np.array([1.0, 1.0], dtype=np.float64))
        self.assertEqual(scaled["x_normalization"], "source_rms")
        self.assertIn(r"\mathrm{rms}", scaled["x_label"])

    def test_reference_rms_replot_uses_saved_reference_scale(self) -> None:
        pdf_result = {
            "pdf_name": "normalized_u12_by_vorticity_rms",
            "source_field": "dux_dy",
            "normalization": "reference_global_rms",
            "bin_edges": np.array([0.0, 1.0, 2.0], dtype=np.float64),
            "bin_centers": np.array([0.5, 1.5], dtype=np.float64),
            "counts": np.array([1, 1], dtype=np.int64),
            "pdf": np.array([0.5, 0.5], dtype=np.float64),
            "raw_x_label": r"$u_{1,2}$",
            "source_field_mean": 1.0,
            "source_field_std": 2.0,
            "source_field_rms": 4.0,
            "normalization_scale": 2.0,
            "normalization_offset": 1.0,
            "normalization_reference_field": "vorticity_magnitude",
            "normalization_reference_label": r"$|\boldsymbol{\omega}|$",
            "normalization_reference_std": 3.0,
            "normalization_reference_rms": 5.0,
            "plot_title": "Velocity Gradient PDF",
        }

        scaled = rescale_field_pdf_for_plot(pdf_result, x_normalization="reference_rms")

        np.testing.assert_allclose(scaled["bin_edges"], np.array([0.2, 0.6, 1.0], dtype=np.float64))
        np.testing.assert_allclose(scaled["bin_centers"], np.array([0.4, 0.8], dtype=np.float64))
        np.testing.assert_allclose(scaled["pdf"], np.array([1.25, 1.25], dtype=np.float64))
        self.assertEqual(scaled["x_normalization"], "reference_rms")
        self.assertIn(r"\boldsymbol{\omega}", scaled["x_label"])

    def test_pdf_integrates_to_one(self) -> None:
        values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
        result = compute_distributed_field_pdf(
            values,
            MPI.COMM_SELF,
            bins=5,
            normalization_scale=1.0,
            pdf_name="test_pdf",
            source_field="test_field",
        )

        self.assertEqual(int(result["total_samples"]), 5)
        self.assertEqual(int(np.sum(result["counts"], dtype=np.int64)), 5)
        self.assertAlmostEqual(float(result["pdf_integral"]), 1.0, places=12)

    def test_smoothed_pdf_preserves_area_and_changes_shape(self) -> None:
        values = np.array([-2.0, -2.0, -2.0, 0.0, 2.0, 2.0], dtype=np.float64)
        result = compute_distributed_field_pdf(
            values,
            MPI.COMM_SELF,
            bins=5,
            normalization_scale=1.0,
            pdf_name="test_pdf",
            source_field="test_field",
        )

        smoothed = smooth_field_pdf_for_plot(
            result,
            smoothing=DEFAULT_FIELD_PDF_SMOOTHING,
            sigma_bins=DEFAULT_FIELD_PDF_SMOOTH_SIGMA_BINS,
        )

        raw_area = float(np.sum(result["pdf"] * np.diff(result["bin_edges"]), dtype=np.float64))
        smoothed_area = float(
            np.sum(smoothed["pdf"] * np.diff(smoothed["bin_edges"]), dtype=np.float64)
        )

        np.testing.assert_allclose(smoothed["bin_edges"], result["bin_edges"])
        np.testing.assert_allclose(smoothed["bin_centers"], result["bin_centers"])
        self.assertAlmostEqual(raw_area, smoothed_area, places=12)
        self.assertTrue(np.all(smoothed["pdf"] >= 0.0))
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(smoothed["pdf"], result["pdf"])

    def test_field_pdf_output_path_supports_smooth_subdirectory(self) -> None:
        path = field_pdf_output_path(
            "data/slice_data/example_slices.h5",
            "normalized_density",
            subdirectory="pdf_smooth",
        )
        self.assertTrue(path.endswith("slice_plots/pdf_smooth/normalized_density_example_slices.pdf"))

    def test_standardization_is_invariant_under_affine_transform(self) -> None:
        values = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float64)
        mean = float(np.mean(values))
        std = float(np.std(values))
        scaled_values = 7.5 * values + 2.25
        scaled_mean = float(np.mean(scaled_values))
        scaled_std = float(np.std(scaled_values))

        result = compute_distributed_field_pdf(
            values,
            MPI.COMM_SELF,
            bins=8,
            normalization_scale=std,
            normalization_offset=mean,
            pdf_name="test_pdf",
            source_field="test_field",
            value_range=(-2.0, 2.0),
        )
        scaled_result = compute_distributed_field_pdf(
            scaled_values,
            MPI.COMM_SELF,
            bins=8,
            normalization_scale=scaled_std,
            normalization_offset=scaled_mean,
            pdf_name="test_pdf",
            source_field="test_field",
            value_range=(-2.0, 2.0),
        )

        np.testing.assert_allclose(result["counts"], scaled_result["counts"])
        np.testing.assert_allclose(result["pdf"], scaled_result["pdf"])
        np.testing.assert_allclose(result["bin_edges"], scaled_result["bin_edges"])

    def test_zero_rms_raises_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "normalization_scale"):
            compute_distributed_field_pdf(
                np.zeros(8, dtype=np.float64),
                MPI.COMM_SELF,
                bins=8,
                normalization_scale=0.0,
                pdf_name="test_pdf",
                source_field="test_field",
            )

    def test_fixed_range_reports_underflow_and_overflow(self) -> None:
        values = np.array([-5.0, -1.0, 0.0, 1.0, 6.0], dtype=np.float64)
        result = compute_distributed_field_pdf(
            values,
            MPI.COMM_SELF,
            bins=4,
            value_range=(-2.0, 2.0),
            normalization_scale=1.0,
            pdf_name="test_pdf",
            source_field="test_field",
        )

        self.assertEqual(int(result["underflow_count"]), 1)
        self.assertEqual(int(result["overflow_count"]), 1)
        self.assertIn("Widen the range", str(result["range_warning"]))


if __name__ == "__main__":
    unittest.main()
