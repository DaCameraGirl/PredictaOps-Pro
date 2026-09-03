"""End-to-end browser test for the critical dashboard journey.

Starts the real FastAPI app as a subprocess, drives it with a headless browser via
Playwright, and checks the page actually renders real data — not just that the
process launched. Run standalone (`python tests/test_e2e_dashboard.py`) or via
pytest; set BASE_URL to point at an already-running instance (e.g. the deployed
app) instead of spawning a local server.
"""
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BASE_URL = os.environ.get("BASE_URL")
PORT = 8931  # distinct from the dev-server default to avoid colliding with a running instance


def _wait_for(url: str, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except Exception:
            time.sleep(0.5)
    return False


@pytest.fixture(scope="module")
def base_url(model_dir, feature_table):
    if BASE_URL:
        yield BASE_URL
        return

    python = sys.executable
    proc = subprocess.Popen(
        [python, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(PORT)],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{PORT}"
    try:
        if not _wait_for(f"{url}/api/health"):
            proc.terminate()
            pytest.fail("local server did not become healthy in time")
        yield url
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_critical_dashboard_journey(base_url):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        pytest.skip("playwright not installed; pip install playwright && playwright install chromium")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        console_errors = []
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda exc: console_errors.append(str(exc)))

        page.goto(base_url)
        page.wait_for_selector(".bearing-row", timeout=15000)

        rows = page.locator(".bearing-row")
        assert rows.count() == 4, "all four bearings must render"

        page.click(".bearing-row >> nth=0")
        page.wait_for_selector("text=SHAP", timeout=10000)

        assert "Predictive Maintenance Studio" in page.title()
        assert console_errors == [], f"unexpected browser console errors: {console_errors}"

        browser.close()


def test_studio_hierarchy_navigation_journey(base_url):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        pytest.skip("playwright not installed; pip install playwright && playwright install chromium")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1366, "height": 1000})
        page.goto(base_url)
        page.wait_for_selector("#studio-nav .studio-nav-node[data-kind='site']", timeout=20000)

        site = page.locator("#studio-nav .studio-nav-node[data-kind='site']").first
        assert site.count() == 1
        site.click()

        page.wait_for_selector("#studio-nav .studio-nav-node[data-kind='asset']", timeout=15000)
        asset = page.locator("#studio-nav .studio-nav-node[data-kind='asset']").first
        assert asset.count() == 1
        asset.click()

        page.wait_for_selector("#studio-nav .studio-nav-node[data-kind='component']", timeout=15000)
        component = page.locator("#studio-nav .studio-nav-node[data-kind='component']").first
        assert component.count() == 1
        component.click()
        page.wait_for_selector("#studio-nav .studio-nav-node[data-kind='sensor']", timeout=15000)

        sensor = page.locator("#studio-nav .studio-nav-node[data-kind='sensor']").first
        assert sensor.count() == 1
        sensor.click()

        detail = page.locator("#studio-detail")
        assert "Sensor" in detail.text_content()
        page.wait_for_selector("text=SHAP", timeout=20000)

        main_detail = page.locator("#detail")
        assert "Recommended action" in main_detail.text_content()
        browser.close()

def test_studio_hierarchy_status_tracks_selected_analytics_health(base_url):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        pytest.skip("playwright not installed; pip install playwright && playwright install chromium")

    def nav_node(page, kind, label):
        return page.locator(f"#studio-nav .studio-nav-node[data-kind='{kind}']", has_text=label).first

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1366, "height": 1000})
        page.goto(base_url)
        page.wait_for_selector("#studio-nav .studio-nav-node[data-kind='asset']", timeout=20000)

        nav_node(page, "asset", "Test 2 Machine").click()
        page.wait_for_selector("#studio-nav .studio-nav-node[data-kind='component']", timeout=15000)
        nav_node(page, "component", "Bearing 1").click()
        page.wait_for_selector("#studio-nav .studio-nav-node[data-kind='sensor']", timeout=15000)
        nav_node(page, "sensor", "sensor_1").click()

        page.wait_for_selector("#detail .callout.critical", timeout=20000)
        page.wait_for_function(
            """
            () => {
              const node = (kind, text) => {
                const selector = `#studio-nav .studio-nav-node[data-kind='${kind}']`;
                return [...document.querySelectorAll(selector)].find((entry) => entry.textContent.includes(text));
              };
              return node("asset", "Test 2 Machine")?.dataset.status === "critical"
                && node("component", "Bearing 1")?.dataset.status === "critical"
                && node("sensor", "sensor_1")?.dataset.status === "critical";
            }
            """,
            timeout=15000,
        )

        operations_text = page.locator("#studio-detail").text_content().lower()
        assert "selected analytics" in operations_text
        assert "critical" in operations_text
        assert "persisted alerts" in operations_text
        assert "no active bindings" in operations_text
        assert "no source records" in operations_text
        assert "no persisted maintenance alerts" in operations_text
        browser.close()

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
