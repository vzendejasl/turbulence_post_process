"""Helpers for auto-discovering sibling scalar sampled-data files."""

from __future__ import annotations

import glob
import os
import re


AUTO_SCALAR_FIELD_NAMES = ("density", "pressure")


def infer_cycle_identifier(path):
    """Infer the cycle identifier from a sampled-data path when available."""
    normalized_path = os.path.abspath(path)
    basename = os.path.basename(normalized_path)
    parent_name = os.path.basename(os.path.dirname(normalized_path))

    for text in (basename, parent_name):
        match = re.search(r"cycle_(\d+)", text, flags=re.IGNORECASE)
        if match:
            return match.group(1)

    match = re.search(r"SampledData(\d+)", basename, flags=re.IGNORECASE)
    if match:
        return match.group(1)

    return None


def scalar_field_requested(explicit_scalar_paths, field_name):
    """Return True when the explicit scalar list already appears to cover one field."""
    needle = str(field_name).strip().lower()
    for path in explicit_scalar_paths or []:
        if needle in os.path.basename(str(path)).lower():
            return True
    return False


def discover_auto_scalar_inputs(primary_input_path, explicit_scalar_paths=None):
    """Return auto-discovered density/pressure scalar TXT/HDF5 siblings.

    Preference order is:
      1. exact cycle-matched TXT
      2. exact cycle-matched HDF5
      3. first TXT match in the directory
      4. first HDF5 match in the directory
    """
    directory = os.path.dirname(os.path.abspath(primary_input_path))
    if not os.path.isdir(directory):
        return []

    cycle_id = infer_cycle_identifier(primary_input_path)
    discovered = []

    for field_name in AUTO_SCALAR_FIELD_NAMES:
        if scalar_field_requested(explicit_scalar_paths, field_name):
            continue

        candidates = []
        if cycle_id is not None:
            candidates.extend(
                [
                    os.path.join(directory, f"{field_name}_sampled_data_uniform_interpolated_cycle_{cycle_id}.txt"),
                    os.path.join(directory, f"{field_name}_sampled_data_uniform_interpolated_cycle_{cycle_id}.h5"),
                ]
            )

        candidates.extend(
            sorted(glob.glob(os.path.join(directory, f"{field_name}_sampled_data_uniform_interpolated_cycle_*.txt")))
        )
        candidates.extend(
            sorted(glob.glob(os.path.join(directory, f"{field_name}_sampled_data_uniform_interpolated_cycle_*.h5")))
        )

        selected = None
        seen = set()
        for candidate in candidates:
            normalized = os.path.abspath(candidate)
            if normalized in seen:
                continue
            seen.add(normalized)
            if os.path.exists(normalized):
                selected = normalized
                break

        if selected is not None:
            discovered.append(selected)

    return discovered
