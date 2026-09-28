"""Backend smoke test - exercises the full pipeline without Streamlit."""
import sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from src import config
from src.preprocessing import load_sample_dataset, preprocess, summarise_dataset
from src.prediction import score_dataframe, explain_rows, get_registry
from src.analytics import supplier_table, kpis, risk_over_time, trend_status, segment_suppliers, group_risk, correlation_matrix
from src.anomaly_detection import detect_anomalies, live_alerts
from src.live_data import LiveSimulator

def run(name, raw):
    print(f"\n=== {name} raw={raw.shape}")
    summ = summarise_dataset(raw)
    print("  profile:", summ.schema.profile, "coverage:", {k: round(v,2) for k,v in summ.schema.coverage.items()}, "missing:", summ.schema.missing_features, "target:", summ.schema.target)
    print("  warnings:", summ.warnings)
    df, schema, report = preprocess(raw)
    print("  preprocess:", report["steps"])
    res = score_dataframe(df, schema)
    print("  method:", res.method, res.model_key, "| news:", res.news_source)
    print("  msgs:", res.messages, "| warn:", res.warnings, "| err:", res.errors)
    if not res.ok:
        return
    sdf = res.df
    st = supplier_table(sdf)
    print("  kpis:", kpis(st, sdf))
    print("  levels:", sdf["Risk_Level"].value_counts().to_dict())
    prof = schema.profile
    if prof:
        seg = segment_suppliers(st, config.PROFILES[prof]["cluster_features"], 4, prof)
        print("  clusters:", seg.labels if seg else None, seg.summary[["Cluster","Suppliers","Avg_Risk_Score","Cluster_Label"]].to_dict("records") if seg else None)
        an = detect_anomalies(sdf, config.PROFILES[prof]["anomaly_features"], 0.03)
        print("  anomalies:", an.total if an else None, "critical:", an.critical if an else None, "observed:", an.suppliers_under_observation if an else None)
        if an and not an.alerts.empty:
            print("  sample alert:", an.alerts.iloc[0][["Supplier_ID","Alert_Type","Severity","Detected","Normal_Range","Status"]].to_dict())
    if res.bundle is not None:
        sid = st.iloc[0]["Supplier_ID"]
        rows = sdf[sdf["Supplier_ID"] == sid].tail(3)
        contrib, base, method = explain_rows(res.bundle, rows)
        print("  explain:", method, "base", base, contrib.head(4).to_dict("records") if contrib is not None else None)
    rt = risk_over_time(sdf)
    print("  trend points:", len(rt), "status:", trend_status(rt["Risk_Score"]) if not rt.empty else None)
    for by in ("Country", "Industry"):
        g = group_risk(st, by)
        if not g.empty: print(f"  by {by}:", len(g))
    cm = correlation_matrix(sdf, config.PROFILES[prof]["features"] if prof else [])
    print("  corr shape:", cm.shape)

t0 = time.time()
reg = get_registry(); print("registry models:", list(reg.bundles), "errors:", reg.load_errors)
run("FINANCIAL CSV", load_sample_dataset("financial"))
run("PROCESSED XLSX", load_sample_dataset("processed"))
run("OPERATIONAL CSV", load_sample_dataset("operational"))
# renamed operational columns + fractions + missing values
op = load_sample_dataset("operational").head(800).rename(columns={"Delivery_Delay_Days":"Delivery delay (days)","Defect_Rate":"defect_rate_pct","Lead_Time_Days":"LeadTime","Order_Fulfillment_Rate":"Fulfilment Rate","Cost_Variation":"cost variance %"})
op.loc[op.sample(50, random_state=1).index, "defect_rate_pct"] = np.nan
op = op.drop(columns=["Performance_Score", "Risk_Event"])
run("RENAMED OPERATIONAL (no label, no perf)", op)
# totally unrelated schema with a label
rng = np.random.default_rng(0)
weird = pd.DataFrame({"vendor": [f"V{i%30}" for i in range(600)], "metric_a": rng.normal(size=600), "metric_b": rng.normal(size=600), "metric_c": rng.normal(size=600), "target": rng.integers(0,2,600)})
run("UNRELATED SCHEMA WITH LABEL", weird)
run("UNRELATED SCHEMA NO LABEL", weird.drop(columns=["target"]))
# empty / all-text
try:
    run("ALL TEXT", pd.DataFrame({"a": ["x","y","z"]*10}))
except Exception as e:
    print("ALL TEXT ->", type(e).__name__, e)

print("\n=== LIVE SIMULATOR")
sim = LiveSimulator(n_suppliers=30, rate=10)
scorer = lambda b: score_dataframe(b).df
sim.start()
for i in range(12):
    n = sim.generate_now(25, scorer, live_alerts)
df = sim.dataframe(); print("  records:", len(df), "cols:", [c for c in df.columns if c.startswith("Risk")], "levels:", df["Risk_Level"].value_counts().to_dict())
print("  history:", len(sim.history), sim.history_frame().tail(1).to_dict("records"))
al = sim.alerts_frame(); print("  alerts:", len(al), al.head(3)[["Icon","Supplier_ID","Alert_Type","Risk_Level","Timestamp","Reason"]].to_dict("records") if not al.empty else None)
print("  last_error:", sim.last_error)
sim.pause(); sim.start(); time.sleep(1.2); print("  advance ->", sim.advance(scorer, live_alerts))
st = supplier_table(df); print("  live kpis:", kpis(st, df))
seg = segment_suppliers(st, config.PROFILES["operational"]["cluster_features"], 4, "operational"); print("  live clusters:", seg.labels if seg else None)
print(f"\nALL DONE in {time.time()-t0:.1f}s")
