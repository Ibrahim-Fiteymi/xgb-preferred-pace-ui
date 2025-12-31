import streamlit as st
import pandas as pd
import joblib
import json

st.set_page_config(page_title="Preferred Pace Predictor", page_icon="🤖", layout="centered")

# -------------------- Theme state --------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def set_theme(t: str):
    st.session_state.theme = t
    st.rerun()

theme = st.session_state.theme

# -------------------- CSS --------------------
if theme == "dark":
    css_vars = """
    :root{
      --page-bg:#0e1117;
      --text:#e6e6e6;
      --muted:rgba(230,230,230,0.65);

      --panel-bg:rgba(255,255,255,0.06);
      --panel-border:rgba(255,255,255,0.14);

      --card-bg:rgba(255,255,255,0.06);
      --card-border:rgba(255,255,255,0.14);

      --badge-bg:rgba(255,255,255,0.06);
      --badge-border:rgba(255,255,255,0.18);

      --advice-bg:rgba(0,128,255,0.10);
      --advice-border:rgba(0,128,255,0.25);

      --plan-bg:rgba(0,200,120,0.10);
      --plan-border:rgba(0,200,120,0.25);

      --why-bg:rgba(255,170,0,0.10);
      --why-border:rgba(255,170,0,0.25);

      --shap-bg:rgba(150,80,255,0.10);
      --shap-border:rgba(150,80,255,0.25);

      --accent:#5aa9ff;
    }
    """
    css_number_fix = """
    /* ✅ REAL FIX: Streamlit number input background is on BaseWeb wrapper, not the <input> */
    div[data-testid="stNumberInput"] div[data-baseweb="input"] > div{
      background: rgba(255,255,255,0.06) !important;
      border-color: rgba(255,255,255,0.14) !important;
      border-radius: 12px !important;
    }
    div[data-testid="stNumberInput"] div[data-baseweb="input"] input{
      background: transparent !important;
      color: #e6e6e6 !important;
      -webkit-text-fill-color: #e6e6e6 !important;
      caret-color: #e6e6e6 !important;
      font-weight: 600 !important;
    }
    div[data-testid="stNumberInput"] div[data-baseweb="input"] input::placeholder{
      color: rgba(230,230,230,0.55) !important;
      -webkit-text-fill-color: rgba(230,230,230,0.55) !important;
    }
    """
else:
    css_vars = """
    :root{
      --page-bg:#ffffff;
      --text:#111111;
      --muted:rgba(0,0,0,0.55);

      --panel-bg:rgba(0,0,0,0.02);
      --panel-border:rgba(0,0,0,0.08);

      --card-bg:#ffffff;
      --card-border:rgba(0,0,0,0.10);

      --badge-bg:rgba(0,0,0,0.03);
      --badge-border:rgba(0,0,0,0.12);

      --advice-bg:rgba(0,128,255,0.06);
      --advice-border:rgba(0,128,255,0.18);

      --plan-bg:rgba(0,200,120,0.06);
      --plan-border:rgba(0,200,120,0.18);

      --why-bg:rgba(255,170,0,0.06);
      --why-border:rgba(255,170,0,0.20);

      --shap-bg:rgba(150,80,255,0.06);
      --shap-border:rgba(150,80,255,0.20);

      --accent:#2563eb;
    }
    """
    css_number_fix = """
    div[data-testid="stNumberInput"] div[data-baseweb="input"] > div{
      background: #ffffff !important;
      border-color: rgba(0,0,0,0.10) !important;
      border-radius: 12px !important;
    }
    div[data-testid="stNumberInput"] div[data-baseweb="input"] input{
      background: transparent !important;
      color: #111111 !important;
      -webkit-text-fill-color: #111111 !important;
      caret-color: #111111 !important;
      font-weight: 600 !important;
    }
    """

st.markdown(
    f"""
<style>
{css_vars}

/* Hide Streamlit chrome */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}

.stApp {{
  background: var(--page-bg);
  color: var(--text);
}}
html, body, [class*="css"] {{
  color: var(--text) !important;
}}

.block-container {{ max-width: 900px; padding-top: 1.5rem; }}

.team-badge {{
    display: inline-block;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    font-weight: 800;
    border: 1px solid var(--badge-border);
    background: var(--badge-bg);
    margin-bottom: 0.75rem;
    letter-spacing: 0.5px;
}}

div[data-testid="stForm"] {{
    background: var(--panel-bg);
    padding: 1.25rem;
    border-radius: 16px;
    border: 1px solid var(--panel-border);
}}

div[data-testid="stMetric"] {{
    background: var(--card-bg);
    padding: 1rem;
    border-radius: 14px;
    border: 1px solid var(--card-border);
}}

.box {{
    padding: 1rem;
    border-radius: 14px;
    border: 1px solid var(--card-border);
    background: var(--card-bg);
}}
.advice-box {{ background: var(--advice-bg); border-color: var(--advice-border); }}
.plan-box   {{ background: var(--plan-bg);   border-color: var(--plan-border);   }}
.why-box    {{ background: var(--why-bg);    border-color: var(--why-border);    }}
.shap-box   {{ background: var(--shap-bg);   border-color: var(--shap-border);   }}

.small-note {{ color: var(--muted); font-size: 0.9rem; }}

/* Inputs & selects */
div[data-baseweb="select"] > div {{
  background: var(--card-bg) !important;
  color: var(--text) !important;
  border-color: var(--card-border) !important;
  border-radius: 12px !important;
}}
div[data-baseweb="select"] span {{
  color: var(--text) !important;
}}

/* Buttons */
div.stButton > button {{
  background: var(--card-bg) !important;
  color: var(--text) !important;
  border: 1px solid var(--card-border) !important;
  border-radius: 12px !important;
  padding: 0.55rem 0.9rem !important;
  font-weight: 700 !important;
}}
div.stButton > button[kind="primary"] {{
  background: var(--accent) !important;
  color: #ffffff !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
}}

/* +/- buttons on number inputs */
div[data-testid="stNumberInput"] button {{
  background: var(--card-bg) !important;
  color: var(--text) !important;
  border: 1px solid var(--card-border) !important;
  border-radius: 10px !important;
}}

{css_number_fix}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------- Helpers --------------------
@st.cache_resource
def load_assets():
    model = joblib.load("xgb_preferred_pace_model.pkl")
    meta = joblib.load("xgb_preferred_pace_meta.pkl")
    opts = json.load(open("ui_options.json", "r", encoding="utf-8"))
    return model, meta, opts

def reset_all():
    keep = {"theme"}
    for k in list(st.session_state.keys()):
        if k not in keep:
            del st.session_state[k]
    st.rerun()

def confidence_label(p_max: float, margin: float) -> str:
    if margin >= 0.25 or p_max >= 0.85:
        return "High"
    if margin >= 0.12 or p_max >= 0.65:
        return "Medium"
    return "Low"

def advice_html(pred_label: str, collaboration: str, content: str, hours: float) -> str:
    pl = str(pred_label).lower()
    coll = str(collaboration).lower()
    cont = str(content).lower()

    items = []
    if "struct" in pl:
        items += [
            "✅ Recommended pace: Structured (fixed weekly plan).",
            "• Use a weekly schedule: topic → practice → short review.",
            "• Use deadlines + checklist. Avoid random studying.",
        ]
    else:
        items += [
            "✅ Recommended pace: Self-paced (flexible plan).",
            "• Study in shorter sessions and adjust as you progress.",
            "• Use milestones: finish unit → quiz → move on.",
        ]

    if "group" in coll:
        items += [
            "👥 Collaboration: Group learning fits you.",
            "• Do 1–2 weekly sessions with 2–4 people (problem solving).",
        ]
    else:
        items += [
            "🧍 Collaboration: Solo learning fits you.",
            "• Focus sessions + notes + self-quizzes (avoid distractions).",
        ]

    if "hand" in cont:
        items += [
            "🛠 Style: Hands-on is best.",
            "• After each topic, do an exercise or mini-project immediately.",
        ]
    else:
        items += [
            "📚 Style: Theory-based is best.",
            "• Outline key concepts first, then solve a small set of questions.",
        ]

    if hours < 6:
        items.append("⏱ Study hours: Low. Increase to ~6–10 hours/week for faster progress.")
    elif hours <= 15:
        items.append("⏱ Study hours: Good. Keep consistency (same days every week).")
    else:
        items.append("⏱ Study hours: High. Add breaks to avoid burnout.")

    li = "".join([f"<li>{x}</li>" for x in items])
    return f"<ul style='margin:0; padding-left: 1.25rem'>{li}</ul>"

def weekly_plan_html(label: str, hours: float) -> str:
    pl = str(label).lower()
    hours = max(1.0, float(hours))

    sessions = 3 if hours < 6 else (4 if hours <= 12 else 5)
    hrs_per = max(1.0, round(hours / sessions, 1))

    if "struct" in pl:
        title = f"{sessions} sessions/week, ~{hrs_per}h each"
        steps = [
            "10 min: quick recap of last session",
            "45–60 min: learn the topic (notes + examples)",
            "45–60 min: practice questions / exercises",
            "10–15 min: summary + checklist for next session",
            "End of week: 1 short quiz + fix mistakes",
        ]
    else:
        title = f"{sessions} flexible blocks/week, ~{hrs_per}h each"
        steps = [
            "Pick a small goal (one lesson / one concept)",
            "Learn it fast (20–30 min)",
            "Apply immediately (30–45 min practice)",
            "Save 5 key notes + 3 mistakes you made",
            "Every 2–3 blocks: one checkpoint quiz to confirm progress",
        ]

    li = "".join([f"<li>{s}</li>" for s in steps])
    return f"<b>🗓 {title}</b><ul style='margin-top:0.5rem; padding-left: 1.25rem'>{li}</ul>"

def why_html(pred_label: str, p0: float, p1: float, collaboration: str, content: str, hours: float, gpa: float) -> str:
    top = max(p0, p1) * 100
    gap = abs(p1 - p0) * 100

    reasons = [
        f"📌 Top probability = {top:.1f}% with a {gap:.1f}% gap between choices.",
        f"📌 Collaboration preference = {collaboration}.",
        f"📌 Preferred content = {content}.",
        f"📌 Weekly study hours = {hours:.1f}.",
    ]

    if gpa <= 1.5:
        reasons.append("📌 Low GPA often benefits from tracking (checklists + deadlines).")
    elif gpa >= 3.5:
        reasons.append("📌 High GPA often correlates with stronger self-regulation.")

    if "struct" in str(pred_label).lower():
        reasons.append("🔁 What-if: If weekly hours drop a lot, the model may shift toward self-paced.")
    else:
        reasons.append("🔁 What-if: Increasing hours + following a schedule may shift toward structured.")

    li = "".join([f"<li>{r}</li>" for r in reasons])
    return f"<ul style='margin:0; padding-left: 1.25rem'>{li}</ul>"

def try_shap_explain(model_obj, x_raw: pd.DataFrame, top_k: int = 5):
    try:
        import numpy as np
        import shap
    except Exception:
        return False, "SHAP is not installed. Run: pip install shap", None

    try:
        if hasattr(model_obj, "named_steps"):
            transformer = None
            for _, step in model_obj.named_steps.items():
                if hasattr(step, "transform"):
                    transformer = step
                    if hasattr(step, "get_feature_names_out"):
                        break

            estimator = model_obj.steps[-1][1] if hasattr(model_obj, "steps") and model_obj.steps else None
            if transformer is None or estimator is None:
                return False, "SHAP could not find a valid transformer/estimator in the pipeline.", None

            X_num = transformer.transform(x_raw)
            try:
                feat_names = [str(f) for f in transformer.get_feature_names_out()]
            except Exception:
                feat_names = [f"f{i}" for i in range(getattr(X_num, "shape", [0, 0])[1])]

            explainer = shap.TreeExplainer(estimator)
            shap_vals = explainer.shap_values(X_num)
            shap_arr = shap_vals[1] if isinstance(shap_vals, list) and len(shap_vals) > 1 else (shap_vals[0] if isinstance(shap_vals, list) else shap_vals)

            sv = np.array(shap_arr)[0]
            abs_sv = np.abs(sv)
            idx = np.argsort(abs_sv)[::-1][:top_k]

            df = pd.DataFrame({
                "feature": [feat_names[i] for i in idx],
                "impact": [float(sv[i]) for i in idx],
                "abs_impact": [float(abs_sv[i]) for i in idx],
            })
            return True, "OK", df

        if any(x_raw.dtypes == "object"):
            return False, "SHAP needs numeric features. Your model likely uses preprocessing in a pipeline.", None

        explainer = shap.TreeExplainer(model_obj)
        shap_vals = explainer.shap_values(x_raw)
        shap_arr = shap_vals[1] if isinstance(shap_vals, list) and len(shap_vals) > 1 else (shap_vals[0] if isinstance(shap_vals, list) else shap_vals)

        sv = __import__("numpy").array(shap_arr)[0]
        abs_sv = __import__("numpy").abs(sv)
        idx = __import__("numpy").argsort(abs_sv)[::-1][:top_k]

        df = pd.DataFrame({
            "feature": [x_raw.columns[i] for i in idx],
            "impact": [float(sv[i]) for i in idx],
            "abs_impact": [float(abs_sv[i]) for i in idx],
        })
        return True, "OK", df

    except Exception as e:
        return False, f"SHAP explanation failed: {e}", None

# -------------------- Header --------------------
st.markdown('<div class="team-badge">THE NO-SLEEP BRIGADE</div>', unsafe_allow_html=True)

top_left, top_right = st.columns([0.72, 0.28])
with top_left:
    st.title("Preferred Pace Prediction (XGBoost)")
    st.caption("Enter student profile and click Predict.")
with top_right:
    cA, cB, cC = st.columns([1, 1, 1])
    with cA:
        if st.button("☀️ Light", type=("primary" if theme == "light" else "secondary"), use_container_width=True):
            set_theme("light")
    with cB:
        if st.button("🌙 Dark", type=("primary" if theme == "dark" else "secondary"), use_container_width=True):
            set_theme("dark")
    with cC:
        if st.button("Reset", use_container_width=True):
            reset_all()

model, meta, opts = load_assets()

# -------------------- Form --------------------
with st.form("predict_form"):
    st.subheader("Student Profile")

    c1, c2, c3 = st.columns(3)
    with c1:
        gpa = st.number_input("GPA (1.00 - 4.00)", min_value=1.0, max_value=4.0, value=3.0, step=0.01, key="gpa")
    with c2:
        weekly_hours_num = st.number_input("Weekly Study Hours (0 - 60)", min_value=0.0, max_value=60.0, value=10.0, step=1.0, key="hours")
    with c3:
        year_num = st.number_input("Year of Study (1 - 4)", min_value=1, max_value=4, value=1, step=1, key="year")

    colA, colB = st.columns(2)
    with colA:
        major = st.selectbox("Major", opts["major"], key="major")
        learning_style = st.selectbox("Learning Style", opts["learning_style"], key="style")
    with colB:
        preferred_content = st.selectbox("Preferred Content", opts["preferred_content"], key="content")
        collaboration_preference = st.selectbox("Collaboration Preference", opts["collaboration_preference"], key="collab")

    submitted = st.form_submit_button("Predict 🚀")

# -------------------- Prediction --------------------
if submitted:
    if weekly_hours_num == 0:
        st.warning("Weekly Study Hours = 0. Prediction may be less reliable. Enter realistic study time if possible.")
    if gpa <= 1.05 or gpa >= 3.95:
        st.warning("GPA is at an extreme value. Interpret the recommendation carefully.")
    if weekly_hours_num >= 50:
        st.info("Very high study hours detected. Add breaks to avoid burnout.")

    x_one = pd.DataFrame([{
        "gpa": gpa,
        "weekly_hours_num": weekly_hours_num,
        "year_num": year_num,
        "major": major,
        "learning_style": learning_style,
        "preferred_content": preferred_content,
        "collaboration_preference": collaboration_preference
    }])

    x_one = x_one.reindex(columns=meta["feature_cols"], fill_value=pd.NA)

    proba = model.predict_proba(x_one)[0]

    le = meta.get("label_encoder", None)
    if le is not None:
        class0 = le.inverse_transform([0])[0]
        class1 = le.inverse_transform([1])[0]
    else:
        class0, class1 = "Class 0", "Class 1"

    p0 = float(proba[0])
    p1 = float(proba[1])

    thr = float(meta.get("best_threshold", 0.5))
    pred = 1 if p1 >= thr else 0
    pred_label = le.inverse_transform([pred])[0] if le is not None else str(pred)

    p0_pct = round(p0 * 100, 1)
    p1_pct = round(p1 * 100, 1)

    margin = abs(p1 - p0)
    conf = confidence_label(max(p0, p1), margin)

    st.success(f"Prediction: **{pred_label}**")

    borderline = margin < 0.10
    if borderline:
        st.warning("Low confidence: probabilities are close. Treat this as a borderline recommendation.")

    m1, m2, m3 = st.columns(3)
    m1.metric(f"P({class1})", f"{p1_pct}%")
    m2.metric(f"P({class0})", f"{p0_pct}%")
    m3.metric("Confidence", conf)
    st.markdown("<div class='small-note'>Confidence is based on the top probability and the gap between the two class probabilities.</div>", unsafe_allow_html=True)

    st.markdown("## Study Advice")
    st.markdown(f"<div class='box advice-box'>{advice_html(pred_label, collaboration_preference, preferred_content, weekly_hours_num)}</div>", unsafe_allow_html=True)

    st.markdown("## Recommended Weekly Plan")
    if borderline:
        tab1, tab2 = st.tabs(["Structured plan", "Self-paced plan"])
        with tab1:
            st.markdown(f"<div class='box plan-box'>{weekly_plan_html('Structured', weekly_hours_num)}</div>", unsafe_allow_html=True)
        with tab2:
            st.markdown(f"<div class='box plan-box'>{weekly_plan_html('Self-paced', weekly_hours_num)}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='box plan-box'>{weekly_plan_html(pred_label, weekly_hours_num)}</div>", unsafe_allow_html=True)

    st.markdown("## Why this recommendation?")
    st.markdown(f"<div class='box why-box'>{why_html(pred_label, p0, p1, collaboration_preference, preferred_content, weekly_hours_num, gpa)}</div>", unsafe_allow_html=True)

    summary_txt = f"""TEAM: THE NO-SLEEP BRIGADE

Prediction: {pred_label}
P({class1}): {p1_pct}%
P({class0}): {p0_pct}%
Confidence: {conf}
Threshold used: {thr:.2f}

Inputs:
- GPA: {gpa}
- Weekly hours: {weekly_hours_num}
- Year: {year_num}
- Major: {major}
- Learning style: {learning_style}
- Preferred content: {preferred_content}
- Collaboration: {collaboration_preference}
"""
    st.download_button(
        "Download recommendation (.txt)",
        data=summary_txt,
        file_name="recommendation.txt",
        mime="text/plain"
    )

    st.markdown("## SHAP Explanation (Bonus)")
    ok, msg, df_top = try_shap_explain(model, x_one, top_k=5)

    if ok and df_top is not None and len(df_top) > 0:
        st.markdown("<div class='box shap-box'><b>Top features affecting this prediction:</b></div>", unsafe_allow_html=True)
        st.dataframe(df_top[["feature", "impact", "abs_impact"]], use_container_width=True)
        st.caption("Impact sign: positive pushes toward the predicted class direction; magnitude shows strength.")
    else:
        st.info(msg)

    with st.expander("Show raw probabilities"):
        st.dataframe(pd.DataFrame([{class0: p0, class1: p1}]), use_container_width=True)
