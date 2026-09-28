"""Capture screenshots of every page (needs a running app + playwright chromium)."""
import sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "screenshots"); OUT.mkdir(parents=True, exist_ok=True)
URL = "http://localhost:8501"
PAGES = ["Overview", "Supplier Analytics", "Risk Prediction", "Anomaly & Alerts", "Live Monitoring"]

def nav(page, name):
    page.locator("section[data-testid='stSidebar'] label").filter(has_text=name).first.click()

def settle(page, t=2.5):
    time.sleep(t)
    try:
        page.wait_for_selector("[data-testid='stStatusWidget']", state="detached", timeout=30000)
    except Exception:
        pass
    time.sleep(0.8)

with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page(viewport={"width": 1500, "height": 950})
    page.goto(URL, wait_until="domcontentloaded")
    page.wait_for_selector("text=SUPPLIER RISK SYSTEM", timeout=60000); settle(page, 4)
    print("title:", page.title())
    page.screenshot(path=str(OUT / "00_empty.png"), full_page=True)
    # load financial sample
    btn = page.get_by_role("button", name="Financial sample (project dataset)")
    if not btn.is_visible():
        page.get_by_text("Sample datasets").click(); time.sleep(0.8)
    btn.click(); settle(page, 10)
    for i, name in enumerate(PAGES[:4], 1):
        nav(page, name); settle(page, 4)
        page.screenshot(path=str(OUT / f"{i:02d}_dataset_{name.split()[0].lower()}.png"), full_page=True)
        print("captured", name)
    # analytics tabs
    nav(page, "Supplier Analytics"); settle(page, 3)
    for tab in ["Segmentation", "Correlation"]:
        page.get_by_role("tab", name=tab).click(); settle(page, 3)
        page.screenshot(path=str(OUT / f"05_analytics_{tab.lower()}.png"), full_page=True)
    # live mode
    nav(page, "Live Data"); settle(page, 3)
    nav(page, "Live Monitoring"); settle(page, 3)
    page.get_by_role("button", name="▶ Start").first.click(); settle(page, 12)
    page.screenshot(path=str(OUT / "06_live_monitoring.png"), full_page=True)
    print("live captured; records text:", page.locator("text=Records generated").first.locator("xpath=..").inner_text()[:60])
    time.sleep(6); page.screenshot(path=str(OUT / "06b_live_monitoring_later.png"), full_page=True)
    nav(page, "Overview"); settle(page, 5)
    page.screenshot(path=str(OUT / "07_live_overview.png"), full_page=True)
    nav(page, "Risk Prediction"); settle(page, 5)
    page.screenshot(path=str(OUT / "08_live_prediction.png"), full_page=True)
    page.get_by_role("button", name="⏸ Pause").first.click(); settle(page, 2)
    # narrow viewport check
    page.set_viewport_size({"width": 1100, "height": 900}); nav(page, "Overview"); settle(page, 4)
    page.screenshot(path=str(OUT / "09_narrow_overview.png"), full_page=True)
    b.close()
print("done ->", OUT)
