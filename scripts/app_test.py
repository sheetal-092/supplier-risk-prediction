"""Headless UI test using streamlit.testing.v1.AppTest."""
import io, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from streamlit.testing.v1 import AppTest
from src.state import PAGES

def check(at, label):
    errs = [e.value for e in at.error if "Something went wrong" in e.value]
    exc = at.exception
    print(f"  [{label}] exc={len(exc)} err={len(errs)} warn={len(at.warning)}")
    for e in exc:
        print("    EXC:", e.value[:400] if hasattr(e, 'value') else e)
        print("    ", getattr(e, 'stack_trace', '')[:1500] if hasattr(e,'stack_trace') else '')
    for e in errs:
        print("    ERR:", e[:300])
    return not exc

def click(at, key):
    at.button(key=key).click().run()

def click_label(at, text):
    for b in at.button:
        if text in b.label:
            b.click(); at.run(); return True
    raise AssertionError(f"button '{text}' not found; have {[b.label for b in at.button]}")

ok = True
at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
at.run()
ok &= check(at, "initial (no data)")
print("  title markdowns:", [m.value[:60] for m in at.markdown[:2]])

# ---- Dataset mode with financial sample
click_label(at, "Financial sample")
ok &= check(at, "financial sample loaded / overview")
print("  KPI metrics markdown count:", len(at.markdown))
for p in PAGES[1:]:
    at.radio(key="page").set_value(p).run()
    ok &= check(at, f"financial -> {p}")

# ---- filters
at.radio(key="page").set_value(PAGES[0]).run()
at.multiselect(key="f_levels").set_value(["HIGH", "MEDIUM"]).run(); ok &= check(at, "filter levels HIGH+MEDIUM")
at.slider(key="f_score").set_value((30, 100)).run(); ok &= check(at, "filter score 30-100")
clusters = at.multiselect(key="f_cluster").options
print("  cluster options:", clusters)
at.multiselect(key="f_cluster").set_value(clusters[:1]).run(); ok &= check(at, "filter cluster")
at.radio(key="page").set_value(PAGES[2]).run(); ok &= check(at, "prediction with filters")
at.radio(key="page").set_value(PAGES[3]).run(); ok &= check(at, "anomaly with filters")
click_label(at, "Reset filters"); ok &= check(at, "reset filters")

# ---- operational sample + processed sample
at.radio(key="page").set_value(PAGES[0]).run()
click_label(at, "Operational sample"); ok &= check(at, "operational sample / overview")
for p in PAGES[1:4]:
    at.radio(key="page").set_value(p).run(); ok &= check(at, f"operational -> {p}")
click_label(at, "Processed dataset"); ok &= check(at, "processed sample")
at.radio(key="page").set_value(PAGES[2]).run(); ok &= check(at, "processed -> prediction")

# ---- analysis settings
at.selectbox[0].set_value("Latest record").run(); ok &= check(at, "agg latest")

# ---- Live mode
at.radio(key="mode").set_value("Live Data").run(); ok &= check(at, "switch to live (no data yet)")
for p in PAGES:
    at.radio(key="page").set_value(p).run(); ok &= check(at, f"live empty -> {p}")
click(at, "sb_start"); ok &= check(at, "start")
time.sleep(1.6); at.run(); ok &= check(at, "after 1.6s (auto fragment)")
print("  sim records:", len(at.session_state["sim"].records), "running:", at.session_state["sim"].running)
time.sleep(1.6); at.run()
for p in PAGES:
    at.radio(key="page").set_value(p).run(); ok &= check(at, f"live running -> {p}")
    time.sleep(0.6)
print("  sim records:", len(at.session_state["sim"].records), "alerts:", len(at.session_state["sim"].alerts), "history:", len(at.session_state["sim"].history))
at.select_slider(key="live_rate").set_value(20).run(); time.sleep(1.2); at.run(); ok &= check(at, "rate 20")
print("  rate:", at.session_state["sim"].rate, "records:", len(at.session_state["sim"].records))
click(at, "sb_pause"); ok &= check(at, "pause"); n1 = len(at.session_state["sim"].records)
time.sleep(1.2); at.run(); n2 = len(at.session_state["sim"].records); print("  paused stable:", n1 == n2, n1, n2, "status", at.session_state["sim"].status)
# filters in live
at.multiselect(key="f_levels").set_value(["HIGH"]).run(); ok &= check(at, "live filter HIGH")
click_label(at, "Reset filters")
click(at, "lv_reset"); ok &= check(at, "reset"); print("  after reset records:", len(at.session_state["sim"].records), "total", at.session_state["sim"].total_generated)
# back to dataset -> data still there?
at.radio(key="mode").set_value("Dataset").run(); ok &= check(at, "back to dataset mode")
print("  dataset still loaded:", at.session_state["dataset"]["name"] if at.session_state["dataset"] else None)

# ---- upload paths (CSV + Excel + bad file) via handle_upload logic
from src.preprocessing import load_uploaded_file, DataValidationError
import pandas as pd
class FakeUpload:
    def __init__(self, name, data): self.name, self._d, self.size = name, data, len(data)
    def getvalue(self): return self._d
df = pd.read_csv(ROOT / "data/sample_operational_supplier_data.csv").head(500)
csv_bytes = df.to_csv(index=False).encode(); print("  CSV upload:", load_uploaded_file(FakeUpload("x.csv", csv_bytes)).shape)
buf = io.BytesIO(); df.to_excel(buf, index=False); print("  XLSX upload:", load_uploaded_file(FakeUpload("x.xlsx", buf.getvalue())).shape)
buf = io.BytesIO()
with pd.ExcelWriter(buf) as w:
    pd.DataFrame({"Item":["a"]}).to_excel(w, sheet_name="README", index=False); df.to_excel(w, sheet_name="Data", index=False)
print("  multi-sheet XLSX picks data sheet:", load_uploaded_file(FakeUpload("m.xlsx", buf.getvalue())).shape)
for name, data in [("empty.csv", b""), ("bad.xlsx", b"not an excel file"), ("junk.csv", b"\x00\x01\x02")]:
    try:
        r = load_uploaded_file(FakeUpload(name, data)); print(f"  {name}: loaded {r.shape}")
    except DataValidationError as e:
        print(f"  {name}: DataValidationError -> {str(e)[:80]}")
print("\nRESULT:", "PASS" if ok else "FAIL")
