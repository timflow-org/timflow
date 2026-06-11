#!/usr/bin/env python3
"""Compile a TOML overview of timflow's user-facing models and elements.

Run from the repository root:
    python utils/compile_timflow_api.py [output.toml]

Default output: timflow_api.toml in the repo root.

The output TOML is intended for the QGIS plugin so it can discover which
timflow classes exist and what constructor arguments they accept.
"""

import importlib
import inspect
import sys
from pathlib import Path

# ── Add repo root to path when running as a standalone script ──────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Source modules to scan ─────────────────────────────────────────────────
# Structure: domain -> category -> list of module dotted-names.
# Only names present in each module's __all__ are included (or all public
# names if __all__ is absent).
SOURCES = {
    "steady": {
        "models": [
            "timflow.steady.model",
        ],
        "elements": [
            "timflow.steady.well",
            "timflow.steady.linesink",
            "timflow.steady.linedoublet",
            "timflow.steady.uflow",
            "timflow.steady.constant",
            "timflow.steady.inhomogeneity",
            "timflow.steady.circareasink",
            "timflow.steady.inhomogeneity1d",
            "timflow.steady.stripareasink",
            "timflow.steady.linesink1d",
            "timflow.steady.linedoublet1d",
        ],
    },
    "transient": {
        "models": [
            "timflow.transient.model",
        ],
        "elements": [
            "timflow.transient.well",
            "timflow.transient.linesink",
            "timflow.transient.linedoublet",
            "timflow.transient.circareasink",
            "timflow.transient.inhom1d",
            "timflow.transient.stripareasink",
            "timflow.transient.linesink1d",
            "timflow.transient.linedoublet1d",
        ],
    },
}

# ── Classes to skip even if listed in __all__ ──────────────────────────────
# Includes internal base classes, purely internal helpers, and deprecated
# aliases that point to the same underlying class.
SKIP = {
    # internal base classes not meant for direct construction
    "Model",
    "WellBase",
    "WellStringBase",
    "LineSinkBase",
    "LineSinkDitchBase",
    "LineSink1DBase",
    "Element",
    "ConstantStar",
    "ConstantInside",
    "WellTest",
    "MScreenLineSink",
    "Xsection",
    "HstarXsection",
    "AreaSinkXsection",
    "FluxDiffLineSink1D",
    "HeadDiffLineSink1D",
    "XsectionAreaSinkInhom",
    # deprecated aliases (use the non-deprecated names instead)
    "ImpLineDoublet",
    "ImpLineDoubletString",
    "LeakyLineDoublet",
    "LeakyLineDoubletString",
    "ImpLineDoublet1D",
    "LeakyLineDoublet1D",
    "HeadLineSink",
    "HeadLineSinkString",
    "HeadLineSinkHo",
    "LineSinkDitch",
    "LineSinkDitchString",
    "HeadLineSink1D",  # deprecated in transient
}


# ── TOML serialisation helpers ─────────────────────────────────────────────


def _toml_scalar(val) -> str:
    """Serialise a scalar Python value to a TOML value string."""
    if val is None:
        return '"null"'
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        # Ensure TOML-valid floats (no bare 'inf', 'nan' edge cases here)
        return repr(val)
    if isinstance(val, str):
        # Escape backslashes and double-quotes
        escaped = val.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(val, (list, tuple)):
        inner = ", ".join(_toml_scalar(v) for v in val)
        return f"[{inner}]"
    # Fallback: store the repr as a string so the TOML stays valid
    escaped = repr(val).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


# ── Signature extraction ───────────────────────────────────────────────────


def get_params(cls):
    """Return (required, optional) constructor parameters for *cls*.

    Parameters
    ----------
    cls : type

    Returns
    -------
    required : list[str]
        Names of parameters that have no default value (excluding ``self``).
    optional : list[tuple[str, any]]
        ``(name, default)`` pairs for parameters that carry a default.
    """
    try:
        sig = inspect.signature(cls.__init__)
    except (ValueError, TypeError):
        return [], []

    required = []
    optional = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if param.default is inspect.Parameter.empty:
            required.append(name)
        else:
            optional.append((name, param.default))

    return required, optional


# ── Module scanning ────────────────────────────────────────────────────────


def scan_module(module_name: str) -> dict:
    """Import *module_name* and return ``{class_name: (required, optional)}``.

    Only classes in ``__all__`` (or public names if ``__all__`` is absent)
    that are not in :data:`SKIP` are returned.
    """
    mod = importlib.import_module(module_name)
    if hasattr(mod, "__all__"):
        candidates = mod.__all__
    else:
        candidates = [n for n in dir(mod) if not n.startswith("_")]

    result = {}
    for name in candidates:
        if name in SKIP:
            continue
        # Auto-skip any class whose name ends with "Base" (internal base classes)
        if name.endswith("Base"):
            continue
        obj = getattr(mod, name, None)
        if obj is None or not inspect.isclass(obj):
            continue
        # Skip classes that are merely re-exported from another module
        if obj.__module__ != module_name:
            continue
        result[name] = get_params(obj)
    return result


# ── TOML builder ──────────────────────────────────────────────────────────


def build_toml(sources: dict) -> str:
    lines = [
        "# timflow API overview",
        "# Auto-generated by utils/compile_timflow_api.py — do not edit by hand.",
        '# Default value "null" means the Python default is None.',
        "",
    ]

    for domain, categories in sources.items():  # "steady" / "transient"
        for category, module_list in categories.items():  # "models" / "elements"
            for module_name in module_list:
                classes = scan_module(module_name)
                if not classes:
                    continue

                for cls_name, (required, optional) in sorted(classes.items()):
                    lines.append(f"[{domain}.{category}.{cls_name}]")
                    lines.append(f'module = "{module_name}"')

                    req_str = "[" + ", ".join(f'"{r}"' for r in required) + "]"
                    lines.append(f"args = {req_str}")

                    for param_name, default in optional:
                        lines.append(f"{param_name} = {_toml_scalar(default)}")

                    lines.append("")

    return "\n".join(lines)


# ── Entry point ───────────────────────────────────────────────────────────


def main():
    if len(sys.argv) > 1:
        output_path = Path(sys.argv[1])
    else:
        output_path = Path(__file__).resolve().parent.parent / "timflow_api.toml"

    toml_content = build_toml(SOURCES)
    output_path.write_text(toml_content, encoding="utf-8")
    print(f"Written {output_path}")


if __name__ == "__main__":
    main()
