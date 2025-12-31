import streamlit as st
import pandas as pd
import joblib
import json

st.set_page_config(page_title="Preferred Pace Predictor", page_icon="🤖", layout="centered")

# -------------------- Styling (Light only) --------------------
st.markdown("""
<style>
.block-container { max-width: 900px; padding-top: 1.5rem; }

.team-badge {
    display: inline-block;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    font-weight: 800;
    border: 1px solid rgba(0,0,0,0.12);
    background: rgba(0,0,0,0.03);
    margin-bottom: 0.75rem;
    letter-spacing: 0.5px;
}

div[data-testid="stForm"] {
    background: rgba(0,0,0,0.02);
    padding: 1.25rem;
    border-radius: 16px;
    border: 1px solid rgba(0,0,0,0.08);
}

div[data-testid="stMetric"] {
    background: white;
    padding: 1rem;
    border-radius: 14px;
    border: 1px solid rgba(0,0,0,0.08);
}

.box {
    padding: 1rem;
    border-radius: 14px;
    border: 1px solid rgba(0,0,0,0.12);
}

.advice-box { background: rgba(0, 128, 255, 0.06); border-color: rgba(0, 128, 255, 0.18); }
.plan-box   { background: rgba(0, 200, 120, 0.06); border-color: rgba(0, 200, 120, 0.18); }
.why-box    { background: rgba(255, 170, 0, 0.06); border-color: rgba(255, 170, 0, 0.20); }
.shap-box   { background: rgba(150, 80, 255, 0.06); border-color: rgba(150, 80, 255, 0.20); }

.small-note { color: rgba(0,0,0,0.55); font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# -------------------- Helpers --------------------
@st.cache_resource
def load_assets():
    model = joblib.load("xgb_preferred_pace_model.pkl")
    meta  = joblib.load("xgb_preferred_pace_meta.pkl")
    opts  = json.load(open("ui_options.json", "r", encoding="utf-8"))
    return model, meta, opts

def reset_all():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

def confidence_label(p_max: float, margin: float) -> str:
    if p_max >= 0.80 and margin >= 0.15:
        return "High"
    if p_max >= 0.65 and margin >= 0.08:
        return "Medium"
    return "Low"

def advice_html(pred_label: str, collaboration: str, content: str, hours: float) -> str:
    pl = str(pred_label).lower()
    coll = str(collaboration).lower()
    cont = str(content).lower()

    items = []

    # Pace
    if "struct" in pl:
        items.append("✅ Recommended pace: Structured (fixed weekly plan).")
        items.append("• Use a weekly schedule: topic → practice → short review.")
        items.append("• Use deadlines + checklist. Avoid random studying.")
    else:
        items.append("✅ Recommended pace: Self-paced (flexible plan).")
        items.append("• Study in shorter sessions and adjust as you progress.")
        items.append("• Use milestones: finish unit → quiz → move on.")

    # Collaboration
    if "group" in coll:
        items.append("👥 Collaboration: Group learning fits you.")
        items.append("• Do 1–2 weekly sessions with 2–4 people (problem solving).")
    else:
        items.append("🧍 Collaboration: Solo learning fits you.")
        items.append("• Focus sessions + notes + self-quizzes (avoid distractions).")

    # Content style
    if "hand" in cont:
        items.append("🛠 Style: Hands-on is best.")
        items.append("• After each topic, do an exercise or mini-project immediately.")
    else:
        items.append("📚 Style: Theory-based is best.")
        items.append("• Outline key concepts first, then solve a small set of questions.")

    # Hours
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

    if hours <= 0:
        hours = 6

    if hours < 6:
        sessions = 3
    elif hours <= 12:
        sessions = 4
    else:
        sessions = 5

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

    reasons = []
    reasons.append(f"📌 Top probability = {top:.1f}% with a {gap:.1f}% gap between choices.")
    reasons.append(f"📌 Collaboration preference = {collaboration}.")
    reasons.append(f"📌 Preferred content = {content}.")
    reasons.append(f"📌 Weekly study hours = {hours:.1f}.")

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
        # Pipeline case
        if hasattr(model_obj, "named_steps"):
            transformer = None
            for _, step in model_obj.named_steps.items():
                if hasattr(step, "transform"):
                    transformer = step
                    if hasattr(step, "get_feature_names_out"):
                        break

            estimator = model_obj.steps[-1][1] if hasattr(model_obj, "steps") and len(model_obj.steps) > 0 else None

            if transformer is None or estimator is None:
                return False, "SHAP could not find a valid transformer/estimator in the pipeline.", None

            X_num = transformer.transform(x_raw)

            try:
                feat_names = transformer.get_feature_names_out()
                feat_names = [str(f) for f in feat_names]
            except Exception:
                feat_names = [f"f{i}" for i in range(getattr(X_num, "shape", [0, 0])[1])]

            explainer = shap.TreeExplainer(estimator)
            shap_vals = explainer.shap_values(X_num)

            if isinstance(shap_vals, list):
                shap_arr = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
            else:
                shap_arr = shap_vals

            sv = np.array(shap_arr)[0]
            abs_sv = np.abs(sv)
            idx = np.argsort(abs_sv)[::-1][:top_k]

            df = pd.DataFrame({
                "feature": [feat_names[i] for i in idx],
                "impact": [float(sv[i]) for i in idx],
                "abs_impact": [float(abs_sv[i]) for i in idx],
            })
            return True, "OK", df

        # Direct numeric model
        if any(x_raw.dtypes == "object"):
            return False, "SHAP needs numeric features. Your model likely uses preprocessing in a pipeline.", None

        import numpy as np
        import shap

        explainer = shap.TreeExplainer(model_obj)
        shap_vals = explainer.shap_values(x_raw)

        if isinstance(shap_vals, list):
            shap_arr = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
        else:
            shap_arr = shap_vals

        sv = np.array(shap_arr)[0]
        abs_sv = np.abs(sv)
        idx = np.argsort(abs_sv)[::-1][:top_k]
        feat_names = list(x_raw.columns)

        df = pd.DataFrame({
            "feature": [feat_names[i] for i in idx],
            "impact": [float(sv[i]) for i in idx],
            "abs_impact": [float(abs_sv[i]) for i in idx],
        })
        return True, "OK", df

    except Exception as e:
        return False, f"SHAP explanation failed: {e}", None

# -------------------- Header --------------------
st.markdown('<div class="team-badge">THE NO-SLEEP BRIGADE</div>', unsafe_allow_html=True)

top_left, top_right = st.columns([0.8, 0.2])
with top_left:
    st.title("Preferred Pace Prediction (XGBoost)")
    st.caption("Enter student profile and click Predict.")
with top_right:
    if st.button("Reset"):
        reset_all()

model, meta, opts = load_assets()

# -------------------- Form --------------------
with st.form("predict_form"):
    st.subheader("Student Profile")

    c1, c2, c3 = st.columns(3)
    with c1:
        gpa = st.number_input("GPA (1.00 - 4.00)", min_value=1.0, max_value=4.0, value=3.0, step=0.01)
    with c2:
        weekly_hours_num = st.number_input("Weekly Study Hours (0 - 60)", min_value=0.0, max_value=60.0, value=10.0, step=1.0)
    with c3:
        year_num = st.number_input("Year of Study (1 - 4)", min_value=1, max_value=4, value=1, step=1)

    colA, colB = st.columns(2)
    with colA:
        major = st.selectbox("Major", opts["major"])
        learning_style = st.selectbox("Learning Style", opts["learning_style"])
    with colB:
        preferred_content = st.selectbox("Preferred Content", opts["preferred_content"])
        collaboration_preference = st.selectbox("Collaboration Preference", opts["collaboration_preference"])

    submitted = st.form_submit_button("Predict 🚀")

# -------------------- Prediction --------------------
if submitted:
    # Guardrails
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

    # Ensure column order matches training
    x_one = x_one.reindex(columns=meta["feature_cols"], fill_value=pd.NA)

    proba = model.predict_proba(x_one)[0]

    le = meta.get("label_encoder", None)
    if le is not None:
        class0 = le.inverse_transform([0])[0]
        class1 = le.inverse_transform([1])[0]
    else:
        class0, class1 = "Self-paced", "Structured"

    p0 = float(proba[0])
    p1 = float(proba[1])

    thr = float(meta.get("best_threshold", 0.5))
    pred = 1 if p1 >= thr else 0
    pred_label = le.inverse_transform([pred])[0] if le is not None else (class1 if pred == 1 else class0)

    p0_pct = round(p0 * 100, 1)
    p1_pct = round(p1 * 100, 1)

    p_max = max(p0, p1)
    margin = abs(p1 - p0)
    conf = confidence_label(p_max, margin)

    st.success(f"Prediction: **{pred_label}**")

    if abs(p1 - p0) < 0.10:
        st.info("Borderline case: both learning paces are plausible for this profile.")
    if conf == "Low":
        st.warning("Low confidence: probabilities are close. Treat this as a borderline recommendation.")

    m1, m2, m3 = st.columns(3)
    m1.metric(f"P({class1})", f"{p1_pct}%")
    m2.metric(f"P({class0})", f"{p0_pct}%")
    m3.metric("Confidence", conf)
    st.markdown("<div class='small-note'>Confidence is based on the top probability and the gap between the two class probabilities.</div>", unsafe_allow_html=True)

    st.markdown("## Study Advice")
    st.markdown(f"<div class='box advice-box'>{advice_html(pred_label, collaboration_preference, preferred_content, weekly_hours_num)}</div>", unsafe_allow_html=True)

    st.markdown("## Recommended Weekly Plan")
    if conf == "Low":
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
