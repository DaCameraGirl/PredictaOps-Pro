import joblib
import numpy as np
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
        if result["shap_unavailable"]:
            pytest.skip(f"SHAP unavailable in this environment: {result.get('shap_error')}")
        assert result["shap_unavailable"] is False
        recon = result["reconciliation"]
        assert recon["within_tolerance"], (
            f"base_value + sum(shap) = {recon['base_value_plus_shap_sum']:.4f} but "
            f"model output = {recon['model_output']:.4f}, off by {recon['difference']:.4f}"
        )


def test_successful_shap_initialization_is_used_when_available(sample_rows, monkeypatch):
    class FakeShapValues:
        def __init__(self, prediction, n_features):
            self.base_values = [prediction]
            self.values = np.zeros((1, n_features))

    class FakeTreeExplainer:
        def __init__(self, model):
            self.model = model
            self.calls = 0

        def __call__(self, x):
            self.calls += 1
            return FakeShapValues(float(self.model.predict(x)[0]), x.shape[1])

    def build_fake_explainer(model):
        return FakeTreeExplainer(model)

    monkeypatch.delenv("PMS_DISABLE_SHAP", raising=False)
    monkeypatch.setattr("explain_bearing._build_shap_explainer", build_fake_explainer)

    explainer = BearingRulExplainer()
    row = sample_rows.iloc[[0]]
    result = explainer.explain(row)

    assert result["shap_unavailable"] is False
    assert result["reconciliation"]["within_tolerance"] is True
    assert explainer._explainer.calls == 1


def test_shap_initialization_failure_falls_back_cleanly(sample_rows, monkeypatch):
    def fail_to_build(_model):
        raise OSError("simulated native loader failure")

    monkeypatch.delenv("PMS_DISABLE_SHAP", raising=False)
    monkeypatch.setattr("explain_bearing._build_shap_explainer", fail_to_build)

    explainer = BearingRulExplainer()
    result = explainer.explain(sample_rows.iloc[[0]])

    assert result["shap_unavailable"] is True
    assert "simulated native loader failure" in result["shap_error"]
    assert result["predicted_rul"] >= 0.0
    assert result["top_contributors"] == []


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
