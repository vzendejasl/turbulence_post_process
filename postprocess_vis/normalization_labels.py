"""Helpers for displaying slice-field normalization in plot labels."""

from __future__ import annotations


def _unwrap_math_label(plot_label):
    """Return a label body plus whether it was wrapped in math-mode dollars."""
    label = str(plot_label).strip()
    if label.startswith("$") and label.endswith("$") and len(label) >= 2:
        return label[1:-1], True
    return label, False


def _wrap_math_label(label_core, is_math):
    """Return a label with its original math-mode wrapper restored."""
    if is_math:
        return f"${label_core}$"
    return label_core


def _value_normalization_expression(value_normalization, base_expression):
    """Return the normalized expression for one saved-value normalization mode."""
    mode = str(value_normalization or "none").strip().lower()
    if mode in {"", "none"}:
        return base_expression
    if mode == "global_rms":
        return rf"{base_expression} / \mathrm{{std}}\left({base_expression}\right)"
    if mode == "global_max":
        return rf"{base_expression} / \max\left({base_expression}\right)"
    return base_expression


def _starred_expression(expression):
    """Return one expression with a superscript star."""
    expression = str(expression).strip()
    if "^*" in expression:
        return expression
    if any(token in expression for token in ("/", r"\mathrm{std}", r"\max", " ")):
        return rf"\left({expression}\right)^*"
    if expression.startswith("|") and expression.endswith("|"):
        return f"{expression[:-1]}^*|"
    return f"{expression}^*"


def format_plot_label(plot_label, value_normalization="none", extra_normalization="none"):
    """Return a display label reflecting saved and optional extra normalization."""
    label_core, is_math = _unwrap_math_label(plot_label)
    label_core = _value_normalization_expression(value_normalization, label_core)

    extra_mode = str(extra_normalization or "none").strip().lower()
    if extra_mode in {"velocity", "vorticity"}:
        label_core = _starred_expression(label_core)

    return _wrap_math_label(label_core, is_math)
