"""Standardized module output contract for AeroNexus.

Every module (M1-M4, Fusion) must return a :class:`ModuleOutput` so callers
can rely on a single, stable shape regardless of which model produced the
result. ``risk_score`` and ``confidence`` are kept strictly separate:
``risk_score`` is the module's assessment of how risky the input is, while
``confidence`` is the module's own certainty about that assessment.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Allowed lifecycle statuses. "unavailable" is used when a module could not
# run (e.g. missing model weights / data) but the caller should still get a
# well-formed response rather than an exception.
STATUS_SUCCESS = "success"
STATUS_UNAVAILABLE = "unavailable"
STATUS_ERROR = "error"
_VALID_STATUSES = (STATUS_SUCCESS, STATUS_UNAVAILABLE, STATUS_ERROR)


class ModuleOutput(BaseModel):
    """Standardized output returned by every AeroNexus module."""

    model_config = ConfigDict(extra="allow")

    module: str = Field(..., description="Name of the module that produced this output.")
    prediction: Any = Field(..., description="Module-specific prediction payload.")
    risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Module risk assessment in [0.0, 1.0].",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Module certainty about its prediction in [0.0, 1.0].",
    )
    status: str = Field(
        ...,
        description="Lifecycle status: success | unavailable | error.",
    )
    model_version: str = Field(..., description="Version identifier of the model used.")
    explanation: List[str] = Field(
        default_factory=list,
        description="Human-readable explanation strings for the prediction.",
    )

    @field_validator("status")
    @classmethod
    def _validate_status(cls, value: str) -> str:
        if value not in _VALID_STATUSES:
            raise ValueError(
                f"status must be one of {_VALID_STATUSES}, got {value!r}"
            )
        return value

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain, JSON-safe dict."""
        return self.model_dump()

    @classmethod
    def unavailable(
        cls,
        module: str,
        model_version: str = "unknown",
        reason: str = "module unavailable",
        prediction: Any = None,
        risk_score: float = 0.0,
        confidence: float = 0.0,
    ) -> "ModuleOutput":
        """Build a well-formed response for a module that could not run."""
        return cls(
            module=module,
            prediction=prediction,
            risk_score=risk_score,
            confidence=confidence,
            status=STATUS_UNAVAILABLE,
            model_version=model_version,
            explanation=[reason],
        )

    @classmethod
    def error(
        cls,
        module: str,
        model_version: str = "unknown",
        reason: str = "module error",
        prediction: Any = None,
        risk_score: float = 0.0,
        confidence: float = 0.0,
    ) -> "ModuleOutput":
        """Build a well-formed response for a module that failed at runtime."""
        return cls(
            module=module,
            prediction=prediction,
            risk_score=risk_score,
            confidence=confidence,
            status=STATUS_ERROR,
            model_version=model_version,
            explanation=[reason],
        )


def validate_output(output: Any) -> ModuleOutput:
    """Coerce a dict or ModuleOutput into a validated :class:`ModuleOutput`.

    Accepts either an existing ``ModuleOutput`` (returned as-is) or a mapping
    produced by ``model_dump`` / JSON. Raises ``TypeError`` if the payload is
    not one of the expected types.
    """
    if isinstance(output, ModuleOutput):
        return output
    if isinstance(output, dict):
        return ModuleOutput(**output)
    raise TypeError(
        f"Cannot validate output of type {type(output).__name__}; expected dict or ModuleOutput."
    )


def serialize_output(output: ModuleOutput) -> Dict[str, Any]:
    """Validate then serialize a module output to a JSON-safe dict."""
    return validate_output(output).to_dict()