"""
ML Engine — SchemaValidator

Prevents schema skew by validating incoming feature data against the
canonical model schema (schema.yaml) before it reaches drift detection
or prediction logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

try:
    import polars as pl
    _POLARS_AVAILABLE = True
except ImportError:
    _POLARS_AVAILABLE = False

_SCHEMA_PATH = Path(__file__).parent / "schema.yaml"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ValidationError:
    field: str
    error_type: str   # missing_field | wrong_type | out_of_range | invalid_category | unexpected_field
    message: str
    record_index: int | None = None

    def __str__(self) -> str:
        loc = f"[record {self.record_index}] " if self.record_index is not None else ""
        return f"{loc}{self.error_type} on '{self.field}': {self.message}"


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)
    validated_count: int = 0

    def raise_if_invalid(self) -> None:
        """Raise ValueError summarising all hard errors."""
        if not self.is_valid:
            summary = "; ".join(str(e) for e in self.errors[:10])
            if len(self.errors) > 10:
                summary += f" … and {len(self.errors) - 10} more"
            raise ValueError(f"Schema validation failed: {summary}")


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class SchemaValidator:
    """
    Loads schema.yaml and exposes validate_input() for pre-inference checks.

    Parameters
    ----------
    schema_path:
        Path to a YAML file following the VigilantMLOps schema format.
        Defaults to the bundled schema.yaml next to this module.
    """

    def __init__(self, schema_path: Path | str | None = None) -> None:
        path = Path(schema_path) if schema_path else _SCHEMA_PATH
        with open(path) as fh:
            raw = yaml.safe_load(fh)

        self._schema = raw
        self._model_meta: dict = raw.get("model", {})
        self._numeric: dict[str, dict] = raw.get("features", {}).get("numeric", {})
        self._categorical: dict[str, dict] = raw.get("features", {}).get("categorical", {})
        self._validation_cfg: dict = raw.get("validation", {})

        self._allow_extra: bool = self._validation_cfg.get("allow_extra_fields", False)
        self._coerce: bool = self._validation_cfg.get("coerce_types", True)
        self._null_strategy: str = self._validation_cfg.get("null_strategy", "fill_zero")
        self._out_of_range: str = self._validation_cfg.get("out_of_range", "warn")

        self._required_fields: set[str] = set(self._numeric) | set(self._categorical)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def feature_names(self) -> list[str]:
        """Ordered list: numeric features first, then categorical."""
        return list(self._numeric) + list(self._categorical)

    @property
    def model_version(self) -> str | None:
        return self._model_meta.get("version")

    @property
    def model_name(self) -> str | None:
        return self._model_meta.get("name")

    def validate_input(
        self,
        data: "list[dict[str, Any]] | dict[str, Any] | pl.DataFrame",
    ) -> ValidationResult:
        """
        Validate one or more inference records against the model schema.

        Accepts:
          - A single feature dict (treated as a batch of 1).
          - A list of feature dicts.
          - A Polars DataFrame (converted to list of dicts internally).

        Returns a ValidationResult — never raises by itself.
        Call result.raise_if_invalid() if you want an exception on failure.
        """
        records = self._normalise(data)
        errors: list[ValidationError] = []
        warnings: list[ValidationError] = []

        if not records:
            errors.append(ValidationError(
                field="<input>",
                error_type="missing_field",
                message="Input is empty — no records to validate.",
            ))
            return ValidationResult(is_valid=False, errors=errors, validated_count=0)

        # Schema-level check (fields present / unexpected) against first record
        first = records[0]
        present = set(first.keys())
        errors += self._check_fields(present)

        # Per-record value checks
        for idx, record in enumerate(records):
            errs, warns = self._check_values(record, idx)
            errors += errs
            warnings += warns

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            validated_count=len(records),
        )

    def validate_dataframe(self, df: "pl.DataFrame") -> ValidationResult:
        """Convenience wrapper — identical to validate_input(df)."""
        return self.validate_input(df)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalise(
        self,
        data: "list[dict[str, Any]] | dict[str, Any] | pl.DataFrame",
    ) -> list[dict[str, Any]]:
        if _POLARS_AVAILABLE and isinstance(data, pl.DataFrame):
            return data.to_dicts()
        if isinstance(data, dict):
            return [data]
        return list(data)

    def _check_fields(self, present: set[str]) -> list[ValidationError]:
        errors: list[ValidationError] = []

        missing = self._required_fields - present
        for feat in sorted(missing):
            errors.append(ValidationError(
                field=feat,
                error_type="missing_field",
                message=f"Required feature '{feat}' is absent from the input.",
            ))

        if not self._allow_extra:
            extra = present - self._required_fields
            for feat in sorted(extra):
                # Extra fields are warnings, not hard errors — they won't break
                # inference but may indicate a schema version mismatch.
                pass  # collected below in _check_values so we have record index

        return errors

    def _check_values(
        self,
        record: dict[str, Any],
        idx: int,
    ) -> tuple[list[ValidationError], list[ValidationError]]:
        errors: list[ValidationError] = []
        warnings: list[ValidationError] = []

        # Unexpected field warnings (per-record so we surface them with index)
        if not self._allow_extra:
            for feat in set(record.keys()) - self._required_fields:
                warnings.append(ValidationError(
                    field=feat,
                    error_type="unexpected_field",
                    message=f"Field '{feat}' is not in the model schema and will be ignored.",
                    record_index=idx,
                ))

        # Numeric features
        for feat, spec in self._numeric.items():
            if feat not in record:
                continue  # already caught by _check_fields

            raw_val = record[feat]
            nullable = spec.get("nullable", False)

            # Handle null / sentinel 'na'
            if raw_val is None or raw_val == "na" or raw_val == "-":
                if self._null_strategy == "reject":
                    errors.append(ValidationError(
                        field=feat,
                        error_type="missing_field",
                        message=f"Null/sentinel value for non-nullable numeric feature '{feat}'.",
                        record_index=idx,
                    ))
                continue  # fill_zero / passthrough — skip range check

            # Type coercion
            val = self._coerce_numeric(feat, raw_val, idx, errors)
            if val is None:
                continue

            # Range check
            low, high = spec.get("min"), spec.get("max")
            if low is not None and val < low:
                issue = ValidationError(
                    field=feat,
                    error_type="out_of_range",
                    message=f"Value {val} < min {low} for feature '{feat}'.",
                    record_index=idx,
                )
                (errors if self._out_of_range == "reject" else warnings).append(issue)
            if high is not None and val > high:
                issue = ValidationError(
                    field=feat,
                    error_type="out_of_range",
                    message=f"Value {val} > max {high} for feature '{feat}'.",
                    record_index=idx,
                )
                (errors if self._out_of_range == "reject" else warnings).append(issue)

        # Categorical features
        for feat, spec in self._categorical.items():
            if feat not in record:
                continue

            val = record[feat]
            if val is None or val == "":
                if self._null_strategy == "reject":
                    errors.append(ValidationError(
                        field=feat,
                        error_type="missing_field",
                        message=f"Null/empty value for categorical feature '{feat}'.",
                        record_index=idx,
                    ))
                continue

            categories: list[str] = spec.get("categories", [])
            if categories and str(val) not in categories:
                errors.append(ValidationError(
                    field=feat,
                    error_type="invalid_category",
                    message=(
                        f"Value '{val}' is not a known category for '{feat}'. "
                        f"Expected one of: {categories}"
                    ),
                    record_index=idx,
                ))

        return errors, warnings

    def _coerce_numeric(
        self,
        feat: str,
        raw: Any,
        idx: int,
        errors: list[ValidationError],
    ) -> float | None:
        if isinstance(raw, (int, float)):
            return float(raw)
        if self._coerce:
            try:
                return float(raw)
            except (TypeError, ValueError):
                pass
        errors.append(ValidationError(
            field=feat,
            error_type="wrong_type",
            message=f"Cannot convert value '{raw}' (type {type(raw).__name__}) to float for feature '{feat}'.",
            record_index=idx,
        ))
        return None
