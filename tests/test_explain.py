import joblib
import pytest

from bearing_data import FEATURE_NAMES
from explain_bearing import BearingRulExplainer


@pytest.fixture(scope="module")
def explainer(model_dir):
    return BearingRulExplainer()


@pytest.fixture
def sample_rows(feature_table):
    return feature_table.sample(n=5, random_state=3)


def test_prediction_is_never_negative(explainer, sample_rows):
    for i in range(len(sample_rows)):
        pred = explainer.predict(sample_rows.iloc[[i]])
        assert pred >= 0.0


def test_shap_reconciles_with_model_output_within_tolerance(explainer, sample_rows):
    for i in range(len(sample_rows)):
        result = explainer.explain(sample_rows.iloc[[i]])
        assert result["shap_unavailable"] is False
        recon = result["reconciliation"]
        assert recon["within_tolerance"], (
            f"base_value + sum(shap) = {recon['base_value_plus_shap_sum']:.4f} but "
            f"model output = {recon['model_output']:.4f}, off by {recon['difference']:.4f}"
        )


def test_shap_feature_names_match_model_input_order(explainer, sample_rows):
    row = sample_rows.iloc[[0]]
    result = explainer.explain(row)
    contributor_features = {c["feature"] for c in result["top_contributors"]}
    assert contributor_features.issubset(set(FEATURE_NAMES))
    assert explainer.feature_cols == FEATURE_NAMES


def test_average_model_output_is_labeled_as_model_average_not_physical_baseline(explainer, sample_rows):
    """Regression guard for the earlier "trained-model baseline" naming mistake:
    the field must be the SHAP expected value (model average), not something that
    could be read as a physical bearing baseline."""
    row = sample_rows.iloc[[0]]
    result = explainer.explain(row)
    assert "average_model_output" in result
    assert "base_value" not in result


def test_reloaded_model_predicts_identically_to_freshly_loaded_model(model_dir, sample_rows):
    model_a = joblib.load(model_dir / "bearing_rul_model.joblib")
    model_b = joblib.load(model_dir / "bearing_rul_model.joblib")
    x = sample_rows[FEATURE_NAMES]
    pred_a = model_a.predict(x)
    pred_b = model_b.predict(x)
    import numpy as np
    np.testing.assert_allclose(pred_a, pred_b)


def test_explain_degrades_gracefully_when_shap_raises(explainer, sample_rows, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("simulated SHAP failure")

    monkeypatch.setattr(explainer, "_explainer", boom)
    row = sample_rows.iloc[[0]]
    result = explainer.explain(row)
    assert result["shap_unavailable"] is True
    assert result["predicted_rul"] >= 0.0
    assert result["top_contributors"] == []
