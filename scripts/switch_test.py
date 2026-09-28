import sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from streamlit.testing.v1 import AppTest
from src.state import PAGES
at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180); at.run()
at.session_state["page"] = PAGES[4]; at.run()
btn = [b for b in at.button if "Switch to Live" in b.label]
print("switch button present:", bool(btn))
btn[0].click(); at.run()
print("mode after click:", at.session_state["mode"], "| exceptions:", len(at.exception), "| errors:", [e.value[:80] for e in at.error])
# empty-state start button on overview in live mode
at.session_state["page"] = PAGES[0]; at.run()
sb = [b for b in at.button if "Start simulation" in b.label]; print("empty-state start button:", bool(sb))
sb[0].click(); at.run(); print("running:", at.session_state["sim"].running, "| exceptions:", len(at.exception))
