"""Tests for the standardized ModuleOutput contract in src/common/schemas.py."""

import pytest

from src.common.schemas import (
    STATUS_ERROR,
    STATUS_SUCCESS,
    STATUS_UNAVAILABLE,
    ModuleOutput,
    serialize_output,
    validate_output,
)


def test_module_output_success():
    out = ModuleOutput(
        module="maintenance",
        prediction={"rul": 120},
        risk_score=0.7,
        confidence=0.85,
        status=STATUS_SUCCESS,
        model_version="rf-v1.0",
        explanation=["high degradation trend"],
    )
    assert out.module == "maintenance"
    assert out.prediction == {"rul": 120}
    assert out.risk_score == 0.7
    assert out.confidence == 0.85
    assert out.status == STATUS_SUCCESS
    assert out.model_version == "rf-v1.0"
    assert out.explanation == ["high degradation trend"]


def test_risk_score_and_confidence_are_separate():
    out = ModuleOutput(
        module="maintenance",
        prediction=None,
        risk_score=0.9,
        confidence=0.1,
        status=STATUS_SUCCESS,
        model_version="v1",
    )
    # risk_score and confidence are independent fields, not aliases of each other
    assert out.risk_score != out.confidence
    assert out.risk_score == 0.9
    assert out.confidence == 0.1


@pytest.mark.parametrize("bad", [-0.001, 1.001, 2.0, -1])
def test_risk_score_out_of_range_rejected(bad):
    with pytest.raises(Exception):
        ModuleOutput(
            module="maintenance",
            prediction=None,
            risk_score=bad,
            confidence=0.5,
            status=STATUS_SUCCESS,
            model_version="v1",
        )


@pytest.mark.parametrize("bad", [-0.001, 1.001, 2.0, -1])
def test_confidence_out_of_range_rejected(bad):
    with pytest.raises(Exception):
        ModuleOutput(
            module="maintenance",
            prediction=None,
            risk_score=0.5,
            confidence=bad,
            status=STATUS_SUCCESS,
            model_version="v1",
        )


@pytest.mark.parametrize("bad", ["ok", "running", "", "SUCCESS"])
def test_invalid_status_rejected(bad):
    with pytest.raises(Exception):
        ModuleOutput(
            module="maintenance",
            prediction=None,
            risk_score=0.5,
            confidence=0.5,
            status=bad,
            model_version="v1",
        )


def test_boundary_scores_accepted():
    out = ModuleOutput(
        module="maintenance",
        prediction=None,
        risk_score=0.0,
        confidence=1.0,
        status=STATUS_SUCCESS,
        model_version="v1",
    )
    assert out.risk_score == 0.0
    assert out.confidence == 1.0


def test_default_explanation_is_empty_list():
    out = ModuleOutput(
        module="maintenance",
        prediction=None,
        risk_score=0.5,
        confidence=0.5,
        status=STATUS_SUCCESS,
        model_version="v1",
    )
    assert out.explanation == []


def test_unavailable_helper():
    out = ModuleOutput.unavailable(
        module="trajectory",
        model_version="none",
        reason="no model weights found",
    )
    assert out.status == STATUS_UNAVAILABLE
    assert out.risk_score == 0.0
    assert out.confidence == 0.0
    assert out.explanation == ["no model weights found"]
    assert out.prediction is None


def test_error_helper():
    out = ModuleOutput.error(
        module="recognition",
        model_version="yolo-v1",
        reason="division by zero",
        risk_score=0.25,
        confidence=0.0,
    )
    assert out.status == STATUS_ERROR
    assert out.risk_score == 0.25
    assert out.confidence == 0.0
    assert out.explanation == ["division by zero"]


def test_to_dict_serializes_all_fields():
    out = ModuleOutput(
        module="fusion",
        prediction={"risk": 0.42},
        risk_score=0.42,
        confidence=0.9,
        status=STATUS_SUCCESS,
        model_version="fusion-v2",
        explanation=["module A high", "module B low"],
    )
    data = out.to_dict()
    assert data == {
        "module": "fusion",
        "prediction": {"risk": 0.42},
        "risk_score": 0.42,
        "confidence": 0.9,
        "status": "success",
        "model_version": "fusion-v2",
        "explanation": ["module A high", "module B low"],
    }


def test_validate_output_accepts_dict_and_module_output():
    out = ModuleOutput(
        module="simulation",
        prediction=0.3,
        risk_score=0.3,
        confidence=0.7,
        status=STATUS_SUCCESS,
        model_version="xgb-v1",
    )
    assert validate_output(out) is out
    rebuilt = validate_output(out.to_dict())
    assert rebuilt == out
    assert isinstance(rebuilt, ModuleOutput)


def test_validate_output_rejects_unknown_type():
    with pytest.raises(TypeError):
        validate_output(123)


def test_validate_output_rejects_missing_field():
    with pytest.raises(Exception):
        validate_output({"module": "maintenance"})


def test_serialize_output_round_trip():
    out = ModuleOutput(
        module="maintenance",
        prediction={"rul": 100.0},
        risk_score=0.55,
        confidence=0.8,
        status=STATUS_SUCCESS,
        model_version="lstm-v1",
        explanation=["trend"],
    )
    data = serialize_output(out)
    assert data["module"] == "maintenance"
    assert data["risk_score"] == 0.55
    assert data["confidence"] == 0.8
    assert data["status"] == "success"
    assert data["model_version"] == "lstm-v1"
    assert data["explanation"] == ["trend"]
    assert data["prediction"] == {"rul": 100.0}


def test_extra_fields_are_preserved():
    out = ModuleOutput(
        module="maintenance",
        prediction=None,
        risk_score=0.1,
        confidence=0.2,
        status=STATUS_SUCCESS,
        model_version="v1",
        engine_id=7,
    )
    assert out.engine_id == 7
    assert out.to_dict()["engine_id"] == 7