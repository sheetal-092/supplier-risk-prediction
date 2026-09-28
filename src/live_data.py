"""Artificial supplier data simulator for **Live Data Mode**.

Every supplier carries a hidden *stress* state that evolves as a mean
reverting random walk with occasional deterioration / recovery episodes.  All
generated operational metrics are functions of that state plus noise, so the
data is internally consistent: a supplier whose delays and defects climb also
becomes more likely to produce a risk event.  The XGBoost *operational* model
in ``models/`` is trained on data drawn from this same generator
(``generate_training_data``), which is why live records can be scored with a
real model instead of hard-coded numbers.

The simulator is deliberately bounded: rolling deques for records, trend
snapshots and alerts, and a hard cap on how many records are created per
refresh, so the Streamlit app never freezes.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Deque, Dict, List, Optional

import numpy as np
import pandas as pd

from src import config

INDUSTRIES = ["Automotive", "Electronics", "Machinery", "Chemicals", "Consumer Goods", "Pharma", "Textiles", "Metals"]
COUNTRIES = ["India", "Germany", "USA", "China", "Japan", "Vietnam", "Mexico", "Poland"]
NAME_PREFIX = ["Nova", "Apex", "Orion", "Vertex", "Summit", "Atlas", "Zenith", "Delta", "Helix", "Quantum", "Pioneer", "Keystone", "Ironclad", "Blue Ridge", "Harbor", "Meridian"]
NAME_SUFFIX = ["Components", "Industries", "Manufacturing", "Logistics", "Supplies", "Systems", "Materials", "Works", "Precision", "Global"]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class SupplierState:
    supplier_id: str
    name: str
    industry: str
    country: str
    base_delay: float
    base_defect: float
    base_lead: float
    base_price: float
    base_qty: float
    stress: float = 0.15
    regime: str = "stable"          # stable | deteriorating | recovering
    regime_age: int = 0

    def step(self, rng: np.random.Generator) -> None:
        """Advance the latent state by one tick."""
        self.regime_age += 1
        if self.regime == "stable":
            self.stress += rng.normal(0, 0.015) + (0.12 - self.stress) * 0.05
            if rng.random() < 0.015:
                self.regime, self.regime_age = "deteriorating", 0
        elif self.regime == "deteriorating":
            self.stress += abs(rng.normal(0.04, 0.02))
            if self.regime_age > 6 and rng.random() < 0.10:
                self.regime, self.regime_age = "recovering", 0
        else:  # recovering
            self.stress -= abs(rng.normal(0.05, 0.02))
            if self.stress < 0.15:
                self.regime, self.regime_age = "stable", 0
        self.stress = float(np.clip(self.stress, 0.0, 1.0))

    def record(self, ts: datetime, rng: np.random.Generator) -> Dict[str, object]:
        s = self.stress
        delay = max(0.0, self.base_delay * (1 + 3.5 * s) + rng.normal(0, 0.8 + 2.5 * s))
        # occasional spikes (late shipments) become likelier under stress
        if rng.random() < 0.03 + 0.15 * s:
            delay += rng.gamma(2.0, 3.0)
        defect = max(0.0, self.base_defect * (1 + 2.5 * s) + rng.normal(0, 0.4 + 0.8 * s))
        lead = max(1.0, self.base_lead * (1 + 0.8 * s) + rng.normal(0, 1.5))
        qty = max(1.0, self.base_qty * (1 - 0.3 * s) * rng.lognormal(0, 0.25))
        cost_var = rng.normal(1.0 + 18.0 * s, 1.5 + 4.0 * s)          # % deviation from contract price
        if rng.random() < 0.02 + 0.08 * s:
            cost_var += rng.normal(10, 5)
        unit_price = self.base_price * (1 + cost_var / 100.0)
        value = qty * unit_price
        fulfil = float(np.clip(100 - 28 * s - 0.5 * delay + rng.normal(0, 2.0), 35, 100))
        quality = float(np.clip(100 - 7.0 * defect - 14 * s + rng.normal(0, 2.5), 0, 100))
        delay_component = (1 - min(delay, 30) / 30) * 100
        cost_component = (1 - min(abs(cost_var), 40) / 40) * 100
        performance = float(np.clip(0.35 * fulfil + 0.35 * quality + 0.15 * delay_component + 0.15 * cost_component, 0, 100))
        # ground-truth risk event probability (used only for model training)
        logit = -5.2 + 6.0 * s + 0.09 * delay + 0.30 * defect + 0.04 * abs(cost_var) - 0.02 * (quality - 80) - 0.03 * (fulfil - 90)
        event = int(rng.random() < _sigmoid(logit))
        return {
            "Supplier_ID": self.supplier_id,
            "Supplier_Name": self.name,
            "Industry": self.industry,
            "Country": self.country,
            "Date": ts,
            "Delivery_Delay_Days": round(delay, 2),
            "Defect_Rate": round(defect, 2),
            "Lead_Time_Days": round(lead, 1),
            "Order_Quantity": int(round(qty)),
            "Order_Value": round(value, 2),
            "Unit_Price": round(unit_price, 2),
            "Quality_Score": round(quality, 1),
            "Cost_Variation": round(cost_var, 2),
            "Order_Fulfillment_Rate": round(fulfil, 1),
            "Performance_Score": round(performance, 1),
            "Risk_Event": event,
        }


def make_suppliers(n: int, seed: int) -> List[SupplierState]:
    rng = np.random.default_rng(seed)
    suppliers: List[SupplierState] = []
    for i in range(n):
        sid = f"S{1000 + i + 1}"
        name = f"{rng.choice(NAME_PREFIX)} {rng.choice(NAME_SUFFIX)}"
        suppliers.append(
            SupplierState(
                supplier_id=sid,
                name=name,
                industry=str(rng.choice(INDUSTRIES)),
                country=str(rng.choice(COUNTRIES)),
                base_delay=float(rng.uniform(0.3, 3.0)),
                base_defect=float(rng.uniform(0.4, 3.0)),
                base_lead=float(rng.uniform(6, 28)),
                base_price=float(rng.uniform(8, 400)),
                base_qty=float(rng.uniform(80, 1500)),
                stress=float(np.clip(rng.beta(2, 8), 0.02, 0.6)),
                regime="deteriorating" if rng.random() < 0.08 else "stable",
            )
        )
    return suppliers


def generate_training_data(n_records: int = 30_000, n_suppliers: int = 120, seed: int = 7) -> pd.DataFrame:
    """Offline draw from the simulator used to train the operational model."""
    rng = np.random.default_rng(seed)
    suppliers = make_suppliers(n_suppliers, seed)
    start = datetime(2024, 1, 1)
    rows: List[Dict[str, object]] = []
    ticks = max(1, n_records // n_suppliers)
    for t in range(ticks):
        ts = start + timedelta(days=3 * t)
        for sup in suppliers:
            sup.step(rng)
            rows.append(sup.record(ts + timedelta(hours=float(rng.integers(0, 24))), rng))
    return pd.DataFrame(rows[:n_records])


# --------------------------------------------------------------------------- #
# Runtime simulator (lives in st.session_state)
# --------------------------------------------------------------------------- #
@dataclass
class LiveSimulator:
    n_suppliers: int = 40
    rate: int = 5                                  # records per second
    seed: int = 42
    running: bool = False
    total_generated: int = 0
    started_at: Optional[float] = None
    _last_tick: Optional[float] = None
    _carry: float = 0.0
    _cursor: int = 0
    suppliers: List[SupplierState] = field(default_factory=list)
    records: Deque[Dict[str, object]] = field(default_factory=lambda: deque(maxlen=config.LIVE_MAX_RECORDS))
    history: Deque[Dict[str, object]] = field(default_factory=lambda: deque(maxlen=config.LIVE_MAX_HISTORY))
    alerts: Deque[Dict[str, object]] = field(default_factory=lambda: deque(maxlen=config.LIVE_MAX_ALERTS))
    last_error: Optional[str] = None
    _rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(42))

    def __post_init__(self) -> None:
        self.reset(keep_config=True)

    # ------------------------------------------------------------------ control
    def reset(self, keep_config: bool = True) -> None:
        self.suppliers = make_suppliers(self.n_suppliers, self.seed)
        self._rng = np.random.default_rng(self.seed + 1)
        self.records = deque(maxlen=config.LIVE_MAX_RECORDS)
        self.history = deque(maxlen=config.LIVE_MAX_HISTORY)
        self.alerts = deque(maxlen=config.LIVE_MAX_ALERTS)
        self.total_generated = 0
        self._carry = 0.0
        self._cursor = 0
        self._last_tick = None
        self.started_at = None
        self.running = False
        self.last_error = None
        self._supplier_prev_risk: Dict[str, float] = {}
        self._alert_cooldown: Dict[str, float] = {}

    def start(self) -> None:
        self.running = True
        self._last_tick = time.time()
        if self.started_at is None:
            self.started_at = time.time()

    def pause(self) -> None:
        self.running = False
        self._last_tick = None

    @property
    def status(self) -> str:
        return "RUNNING" if self.running else "PAUSED"

    # --------------------------------------------------------------- generation
    def _generate(self, n: int, ts: datetime) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []
        for i in range(n):
            sup = self.suppliers[self._cursor % len(self.suppliers)]
            self._cursor += 1
            if self._cursor % len(self.suppliers) == 0:
                for s in self.suppliers:
                    s.step(self._rng)
            rows.append(sup.record(ts + timedelta(milliseconds=int(i * 1000 / max(n, 1))), self._rng))
        return pd.DataFrame(rows)

    def advance(self, scorer: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
                alert_fn: Optional[Callable[["LiveSimulator", pd.DataFrame], List[Dict[str, object]]]] = None) -> int:
        """Generate the records owed since the last tick (bounded).

        ``scorer`` receives the raw batch and returns it with risk columns
        appended (the same XGBoost pipeline used in Dataset Mode).
        """
        if not self.running:
            return 0
        now = time.time()
        if self._last_tick is None:
            self._last_tick = now
            return 0
        elapsed = now - self._last_tick
        self._last_tick = now
        owed = elapsed * self.rate + self._carry
        n = int(owed)
        self._carry = owed - n
        n = min(n, config.LIVE_MAX_BATCH)
        if n <= 0:
            return 0
        return self.generate_now(n, scorer, alert_fn)

    def generate_now(self, n: int, scorer=None, alert_fn=None) -> int:
        batch = self._generate(n, datetime.now())
        if scorer is not None:
            try:
                batch = scorer(batch)
                self.last_error = None
            except Exception as exc:  # noqa: BLE001
                self.last_error = f"Scoring failed: {exc}"
        for row in batch.to_dict("records"):
            self.records.append(row)
        self.total_generated += len(batch)
        self._snapshot(batch)
        if alert_fn is not None and "Risk_Score" in batch.columns:
            try:
                for alert in alert_fn(self, batch):
                    self.alerts.appendleft(alert)
            except Exception as exc:  # noqa: BLE001
                self.last_error = f"Alert generation failed: {exc}"
        return len(batch)

    def _snapshot(self, batch: pd.DataFrame) -> None:
        if "Risk_Score" not in batch.columns:
            return
        current = self.supplier_latest()
        counts = current["Risk_Level"].value_counts() if not current.empty else pd.Series(dtype=int)
        self.history.append(
            {
                "Time": datetime.now(),
                "Avg_Risk_Score": float(batch["Risk_Score"].mean()),
                "Portfolio_Avg_Risk": float(current["Risk_Score"].mean()) if not current.empty else float("nan"),
                "High": int(counts.get("HIGH", 0)),
                "Medium": int(counts.get("MEDIUM", 0)),
                "Low": int(counts.get("LOW", 0)),
                "Records": self.total_generated,
            }
        )

    # ------------------------------------------------------------------- views
    def dataframe(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame()
        df = pd.DataFrame(list(self.records))
        df["Date"] = pd.to_datetime(df["Date"])
        return df

    def history_frame(self) -> pd.DataFrame:
        return pd.DataFrame(list(self.history)) if self.history else pd.DataFrame()

    def alerts_frame(self) -> pd.DataFrame:
        return pd.DataFrame(list(self.alerts)) if self.alerts else pd.DataFrame()

    def supplier_latest(self) -> pd.DataFrame:
        df = self.dataframe()
        if df.empty or "Risk_Score" not in df.columns:
            return pd.DataFrame()
        return df.sort_values("Date").groupby("Supplier_ID", as_index=False).tail(1)

    def previous_risk(self, supplier_id: str) -> Optional[float]:
        return self._supplier_prev_risk.get(supplier_id)

    def remember_risk(self, supplier_id: str, value: float) -> None:
        self._supplier_prev_risk[supplier_id] = value

    def cooldown_ok(self, key: str, seconds: float = 45.0) -> bool:
        now = time.time()
        last = self._alert_cooldown.get(key, 0.0)
        if now - last >= seconds:
            self._alert_cooldown[key] = now
            return True
        return False
