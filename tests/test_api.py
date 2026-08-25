import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(model_dir, feature_table):
    import main  # imports app/main.py; requires processed features + trained model on disk

    return TestClient(main.app)


def test_health_endpoint(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_only_bearing_1_ever_carries_true_rul(client):
    resp = client.get("/api/snapshot/930")
    assert resp.status_code == 200
    for b in resp.json()["bearings"]:
        if b["bearing"] == "bearing_1":
            assert "true_rul" in b
            assert b["has_ground_truth"] is True
        else:
            assert "true_rul" not in b, f"{b['bearing']} never failed — must not expose a ground-truth RUL"
            assert b["has_ground_truth"] is False


def test_interval_is_explicitly_labeled_as_not_calibrated(client):
    resp = client.get("/api/snapshot/930/bearing/bearing_1")
    note = resp.json()["interval_80"]["note"]
    assert "not a conditionally calibrated guarantee" in note
    assert "walk-forward residuals" in note


def test_health_state_is_one_of_the_defensible_states(client):
    resp = client.get("/api/snapshot/930")
    allowed = {"healthy", "watch", "warning", "critical", "insufficient_evidence"}
    for b in resp.json()["bearings"]:
        assert b["health_state"] in allowed


def test_recommendation_always_requires_human_verification(client):
    resp = client.get("/api/snapshot/930/bearing/bearing_1")
    rec = resp.json()["recommendation"]
    assert rec["requires_human_verification"] is True
    assert "human verification" in rec["disclaimer"]


def test_predicted_rul_is_never_negative_across_the_full_timeline(client):
    resp = client.get("/api/timeline")
    n = resp.json()["n_snapshots"]
    for index in (0, n // 4, n // 2, 3 * n // 4, n - 1):
        snap = client.get(f"/api/snapshot/{index}")
        for b in snap.json()["bearings"]:
            assert b["predicted_rul"] >= 0.0


def test_out_of_range_snapshot_index_is_a_clean_404(client):
    resp = client.get("/api/timeline")
    n = resp.json()["n_snapshots"]
    assert client.get(f"/api/snapshot/{n + 100}").status_code == 404
    assert client.get("/api/snapshot/-1").status_code == 404


def test_unknown_bearing_id_is_a_clean_404(client):
    assert client.get("/api/snapshot/930/bearing/bearing_99").status_code == 404


def test_profile_exposes_scope_statement_and_backtest_metrics(client):
    resp = client.get("/api/profile")
    body = resp.json()
    assert "scope_statement" in body["model"]
    assert "confirmed failure trajectory" in body["model"]["scope_statement"]
    assert body["model"]["validation_method"].startswith("chronological walk-forward")


def test_trajectory_csv_export_matches_dashboard_feature_values(client):
    detail = client.get("/api/snapshot/930/bearing/bearing_1").json()
    csv_resp = client.get("/api/export/trajectory/bearing_1.csv")
    assert csv_resp.status_code == 200
    assert csv_resp.headers["content-type"].startswith("text/csv")
    body = csv_resp.text
    assert detail["timestamp"] in body


def test_snapshot_and_bearing_detail_report_the_same_predicted_rul(client):
    """Guards against the two endpoints silently drifting apart, which happened once
    before when true_rul lookup was only wired into one of them."""
    list_resp = client.get("/api/snapshot/930").json()
    list_value = next(b for b in list_resp["bearings"] if b["bearing"] == "bearing_1")["predicted_rul"]
    detail_resp = client.get("/api/snapshot/930/bearing/bearing_1").json()
    assert detail_resp["predicted_rul"] == list_value


def test_api_starts_when_shap_initialization_fails(monkeypatch):
    import main

    import explain_bearing

    def fail_to_build(_model):
        raise OSError("simulated SHAP import failure")

    monkeypatch.setattr(explain_bearing, "_build_shap_explainer", fail_to_build)
    reloaded_main = importlib.reload(main)
    local_client = TestClient(reloaded_main.app)

    assert local_client.get("/api/health").status_code == 200
    detail = local_client.get("/api/snapshot/930/bearing/bearing_1").json()
    assert detail["shap_unavailable"] is True
    assert "simulated SHAP import failure" in detail["shap_error"]
