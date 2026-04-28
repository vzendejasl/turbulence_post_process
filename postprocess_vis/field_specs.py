"""Field registry helpers for slice rendering."""

from __future__ import annotations

DENSITY_DATASET_NAME = "density"
PRESSURE_DATASET_NAME = "pressure"
DENSITY_GRADIENT_FIELD_NAME = "density_gradient_magnitude"
DENSITY_GRADIENT_FIELD_SPEC = (
    DENSITY_GRADIENT_FIELD_NAME,
    DENSITY_GRADIENT_FIELD_NAME,
    r"$|\nabla \rho|$",
    "density_gradient",
)
SOUND_SPEED_FIELD_NAME = "sound_speed"
MACH_NUMBER_FIELD_NAME = "mach_number"
THERMO_DERIVED_FIELD_NAMES = (
    SOUND_SPEED_FIELD_NAME,
    MACH_NUMBER_FIELD_NAME,
)
THERMO_DERIVED_FIELD_SPECS = {
    SOUND_SPEED_FIELD_NAME: (SOUND_SPEED_FIELD_NAME, SOUND_SPEED_FIELD_NAME, r"$c$", "thermo"),
    MACH_NUMBER_FIELD_NAME: (MACH_NUMBER_FIELD_NAME, MACH_NUMBER_FIELD_NAME, r"$M$", "thermo"),
}

VELOCITY_DATASET_NAMES = {"vx", "vy", "vz"}

BUILTIN_FIELD_MAP = {
    "velocity_magnitude": ("velocity_magnitude", "velocity_magnitude", r"$|\mathbf{u}|$", "velocity"),
    "vx": ("vx", "vx", r"$u_1$", "velocity"),
    "vy": ("vy", "vy", r"$u_2$", "velocity"),
    "vz": ("vz", "vz", r"$u_3$", "velocity"),
    "vorticity_magnitude": ("vorticity_magnitude", "vorticity_magnitude", r"$|\boldsymbol{\omega}|$", "vorticity"),
    "wx": ("omega_x", "wx", r"$\omega_1$", "vorticity"),
    "wy": ("omega_y", "wy", r"$\omega_2$", "vorticity"),
    "wz": ("omega_z", "wz", r"$\omega_3$", "vorticity"),
    "div_u": ("div_u", "div_u", r"$\theta$", "divergence"),
    "q_criterion": ("q_criterion", "q_criterion", r"$Q$", "qcriterion"),
    "r_criterion": ("r_criterion", "r_criterion", r"$R$", "rcriterion"),
}

DERIVED_FIELD_FAMILIES = {"vorticity", "divergence", "qcriterion", "rcriterion", "density_gradient", "thermo"}
DERIVED_DATASET_NAMES = {
    "vorticity_magnitude",
    "omega_x",
    "omega_y",
    "omega_z",
    "div_u",
    "q_criterion",
    "r_criterion",
    DENSITY_GRADIENT_FIELD_NAME,
    *THERMO_DERIVED_FIELD_NAMES,
}


def build_available_field_specs(fields_group):
    """Return the built-in and dataset-backed field specs for one HDF5 fields group."""
    field_specs = dict(BUILTIN_FIELD_MAP)
    dataset_names = list(fields_group.keys())

    for dataset_name in sorted(dataset_names):
        if dataset_name in VELOCITY_DATASET_NAMES:
            continue
        if dataset_name in field_specs:
            continue
        dataset = fields_group[dataset_name]
        display_name = str(dataset.attrs.get("display_name", dataset_name))
        plot_label = str(dataset.attrs.get("plot_label", display_name))
        field_specs[dataset_name] = (dataset_name, dataset_name, plot_label, "scalar")

    if DENSITY_DATASET_NAME in dataset_names:
        field_specs.setdefault(DENSITY_GRADIENT_FIELD_NAME, DENSITY_GRADIENT_FIELD_SPEC)
    if DENSITY_DATASET_NAME in dataset_names and PRESSURE_DATASET_NAME in dataset_names:
        for field_name, field_spec in THERMO_DERIVED_FIELD_SPECS.items():
            field_specs.setdefault(field_name, field_spec)

    return field_specs


def default_requested_field_names(field_lookup):
    """Return the default field selection for one available-field map."""
    requested_fields = ["velocity_magnitude", "vorticity_magnitude", "div_u", "q_criterion", "r_criterion"]
    if DENSITY_GRADIENT_FIELD_NAME in field_lookup:
        requested_fields.append(DENSITY_GRADIENT_FIELD_NAME)
    if MACH_NUMBER_FIELD_NAME in field_lookup:
        requested_fields.append(MACH_NUMBER_FIELD_NAME)
    scalar_fields = [name for name, spec in field_lookup.items() if spec[3] == "scalar"]
    requested_fields.extend(scalar_fields)
    return requested_fields


def finalize_requested_field_names(field_lookup, requested_fields):
    """Return the final field list after applying auto-included derived fields."""
    if requested_fields:
        resolved_fields = list(requested_fields)
        if "q_criterion" in resolved_fields and "r_criterion" not in resolved_fields:
            q_index = resolved_fields.index("q_criterion")
            resolved_fields.insert(q_index + 1, "r_criterion")
        if (
            DENSITY_GRADIENT_FIELD_NAME in field_lookup
            and DENSITY_GRADIENT_FIELD_NAME not in resolved_fields
        ):
            resolved_fields.append(DENSITY_GRADIENT_FIELD_NAME)
        return resolved_fields
    return default_requested_field_names(field_lookup)
