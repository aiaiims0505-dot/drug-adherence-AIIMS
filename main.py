import json
from datetime import date, timedelta
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader

from auth import (
    clear_session_cookie, hash_password, make_session_token,
    read_session, set_session_cookie, verify_password,
)
from content import ARTICLES, get_article, get_by_category
from db import (
    get_active_medications, get_all_patients, get_assessments,
    get_db, get_dose_grid, get_latest_assessment, get_lifestyle_history,
    get_lifestyle_log, get_lifestyle_range, get_sms_log, get_today_logs,
    get_user, get_user_by_email, get_weekly_stats, init_db,
)
from predict import predict

app = FastAPI(title="DA-AIIMS")

BASE = Path(__file__).parent
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")

_jinja_env = Environment(
    loader=FileSystemLoader(str(BASE / "templates")),
    autoescape=True,
    cache_size=0,
)

VAPID_PUBLIC_KEY = "BLc2Vh6tV8HBFQ8bFQ8bFQ8bFQ8bFQ8bFQ8bFQ8bFQ8bFQ8bFQ8bFQ8bFQ8bFQ="


@app.on_event("startup")
async def startup():
    init_db()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _render(template_name: str, **ctx) -> HTMLResponse:
    tmpl = _jinja_env.get_template(template_name)
    return HTMLResponse(tmpl.render(**ctx))


def _redirect(path: str, flash_msg: str = "", flash_type: str = "success") -> RedirectResponse:
    r = RedirectResponse(path, status_code=303)
    if flash_msg:
        r.set_cookie("flash_msg", flash_msg, max_age=5, httponly=False)
        r.set_cookie("flash_type", flash_type, max_age=5, httponly=False)
    return r


def _get_flash(request: Request) -> tuple[str, str]:
    msg = request.cookies.get("flash_msg", "")
    typ = request.cookies.get("flash_type", "success")
    return msg, typ


def _clear_flash(response: HTMLResponse) -> None:
    response.delete_cookie("flash_msg")
    response.delete_cookie("flash_type")


def _today() -> str:
    return str(date.today())


def _week_start() -> str:
    return str(date.today() - timedelta(days=6))


def _calc_streak(user_id: int) -> int:
    history = get_lifestyle_history(user_id, days=30)
    streak = 0
    check = date.today()
    dated = {e["date"]: e for e in history}
    while str(check) in dated:
        streak += 1
        check -= timedelta(days=1)
    return streak


def _patient_required(request: Request):
    sess = read_session(request)
    if not sess or sess.get("role") != "patient":
        return None
    return get_user(sess["id"])


def _doctor_required(request: Request):
    sess = read_session(request)
    if not sess or sess.get("role") != "doctor":
        return None
    return get_user(sess["id"])


def _render_patient(tpl: str, user: dict, request: Request, **ctx) -> HTMLResponse:
    flash_msg, flash_type = _get_flash(request)
    meds = get_active_medications(user["id"])
    today_logs = get_today_logs(user["id"], _today())
    today_pending = sum(1 for m in meds if today_logs.get(m["id"]) not in ("taken", "missed"))
    today_taken = sum(1 for m in meds if today_logs.get(m["id"]) == "taken")
    today_total = len(meds)
    resp = _render(
        tpl,
        user=user,
        today=_today(),
        today_pending=today_pending,
        today_taken=today_taken,
        today_total=today_total,
        flash_msg=flash_msg,
        flash_type=flash_type,
        **ctx,
    )
    _clear_flash(resp)
    return resp


def _render_doctor(tpl: str, user: dict, request: Request, **ctx) -> HTMLResponse:
    flash_msg, flash_type = _get_flash(request)
    resp = _render(tpl, user=user, flash_msg=flash_msg, flash_type=flash_type, **ctx)
    _clear_flash(resp)
    return resp


def _risk_tier(p_score) -> str | None:
    if p_score is None:
        return None
    if p_score > 55.0:
        return "HIGH"
    if p_score >= 47.0:
        return "MODERATE"
    return "LOW"


def _verdict(p_score) -> str | None:
    if p_score is None:
        return None
    if p_score > 55.0:
        return "Non-Adherent"
    if p_score >= 47.0:
        return "Probably Non-Adherent"
    if p_score >= 40.0:
        return "Probably Adherent"
    return "Adherent"


def _factor_category(feature: str) -> str:
    if feature.startswith(("Drug_1_Group_", "Drug_2_Group_", "Drug_3_Group_")):
        return "Medications"
    if feature.startswith(("Occupation_", "Marital_status_")):
        return "Demographics"
    return _FEAT_CATEGORY.get(feature, "Other")


def _factor_reason(factor: dict, form_data: dict) -> str:
    feature = factor.get("feature", "")
    direction = factor.get("patient_direction", 0)
    higher_lower = "above" if direction > 0 else "below"

    value_reasons = {
        "Last_RBS_value": f"Latest random blood sugar entered: {form_data.get('rbs') or 'not provided'} mg/dL.",
        "Last_HBAIC_value": f"Latest HbA1c entered: {form_data.get('hba1c') or 'not provided'}%.",
        "Last_FBS_value": f"Latest fasting blood sugar entered: {form_data.get('fbs') or 'not provided'} mg/dL.",
        "Financial_Stress_Ratio": (
            f"Diabetes drug expense score {form_data.get('drug_cost_score')} compared with "
            f"income score {form_data.get('income_score')}."
        ),
        "Income_Score": f"Monthly income score selected: {form_data.get('income_score')}.",
        "Drug_Cost_Score": f"Monthly diabetes drug expense score selected: {form_data.get('drug_cost_score')}.",
        "Total_Pills": f"Pill burden entered: {form_data.get('total_pills')} pill(s) per day.",
        "Total_Daily_Frequency": f"Pill burden entered: {form_data.get('total_pills')} pill(s) per day.",
        "Total_Drugs_Prescribed": (
            "Diabetes drug classes selected: "
            + ", ".join(
                d for d in [
                    form_data.get("drug1_class"),
                    form_data.get("drug2_class"),
                    form_data.get("drug3_class"),
                ]
                if d and d != "None"
            )
            + "."
        ),
        "Duration_of_illness_months": f"Diabetes duration entered: {form_data.get('duration_months')} month(s).",
        "Duration_Years": f"Diabetes duration entered: {form_data.get('duration_months')} month(s).",
        "Distance_nearby_facility_from_house_approx_km": f"Clinic distance entered: {form_data.get('distance_km')} km.",
        "Time_since_last_lab_investigation_months": (
            f"Time since last lab investigation: {form_data.get('time_since_lab_months') or 'not provided'} month(s)."
        ),
        "Self_Glucose_Monitoring_Monthly": (
            f"Self blood glucose monitoring entered: {form_data.get('self_glucose_monitoring')} time(s) per month."
        ),
        "Comorbidity_Count": f"Other comorbidities selected: {form_data.get('comorbidity_count')}.",
        "Has_HTN": "Hypertension was marked as present." if form_data.get("has_htn") == "1" else "Hypertension was not marked.",
        "On_diabetic_diet": "Diabetic diet marked yes." if form_data.get("on_diabetic_diet") == "1" else "Diabetic diet marked no.",
        "Do_you_exercise": "Regular exercise marked yes." if form_data.get("exercises") == "1" else "Regular exercise marked no.",
        "Do_you_sleep_adequately": "Adequate sleep marked yes." if form_data.get("sleeps_adequately") == "1" else "Adequate sleep marked no.",
        "Currently_any_smoking_alcohol_intake": (
            "Current smoking/alcohol use marked yes."
            if form_data.get("smoking_alcohol") == "1"
            else "Current smoking/alcohol use marked no."
        ),
        "Do_you_get_counselling_regarding_disease_drug_use_drug_adherance_treating_doctor": (
            "Medication counselling from treating doctor marked yes."
            if form_data.get("counselling_doctor") == "1"
            else "Medication counselling from treating doctor marked no."
        ),
        "Rough_monthly_expenses_on_consultation_rupees_250": (
            "Monthly consultation expense of Rs 250 or more marked yes."
            if form_data.get("consult_over_250") == "1"
            else "Monthly consultation expense of Rs 250 or more marked no."
        ),
    }

    if feature in value_reasons:
        return value_reasons[feature]
    if feature.startswith(("Drug_1_Group_", "Drug_2_Group_", "Drug_3_Group_")):
        return "Selected diabetes medication class contributed to the model score."
    return f"This value is {higher_lower} the model training average for this feature."


_FEAT_CATEGORY: dict[str, str] = {
    "Financial_Stress_Ratio": "Finance",
    "Income_Score": "Finance",
    "Drug_Cost_Score": "Finance",
    "Rough_monthly_expenses_on_consultation_rupees_250": "Finance",
    "Last_RBS_value": "Diabetes Control",
    "Last_HBAIC_value": "Diabetes Control",
    "Last_FBS_value": "Diabetes Control",
    "Duration_of_illness_months": "Diabetes Control",
    "Duration_Years": "Diabetes Control",
    "Comorbidity_Count": "Diabetes Control",
    "Has_HTN": "Diabetes Control",
    "Total_Pills": "Medications",
    "Total_Daily_Frequency": "Medications",
    "Total_Drugs_Prescribed": "Medications",
    "Any_new_drug_added_in_the_last_visit": "Medications",
    "Is_the_new_drug_easily_available": "Medications",
    "On_diabetic_diet": "Lifestyle",
    "Do_you_exercise": "Lifestyle",
    "Do_you_sleep_adequately": "Lifestyle",
    "Currently_any_smoking_alcohol_intake": "Lifestyle",
    "Self_Glucose_Monitoring_Monthly": "Lifestyle",
    "Distance_nearby_facility_from_house_approx_km": "Access & Support",
    "Time_since_last_lab_investigation_months": "Access & Support",
    "Do_you_get_counselling_regarding_disease_drug_use_drug_adherance_others": "Access & Support",
    "Do_you_get_counselling_regarding_disease_drug_use_drug_adherance_treating_doctor": "Access & Support",
    "Residence_tribal": "Access & Support",
    "Residence_urban": "Access & Support",
    "Age": "Demographics",
    "Sex": "Demographics",
    "Education": "Demographics",
    "Family_Size": "Demographics",
    "weight_kg": "Demographics",
    "Height_cm": "Demographics",
    "BMI_kg_m2": "Demographics",
}


def _categorize_factors(factors: list) -> dict:
    cats: dict[str, list] = {}
    for f in factors:
        cat = f.get("category") or _factor_category(f.get("feature", ""))
        if cat not in cats:
            cats[cat] = []
        cats[cat].append(f)
    return cats


def _simulate_sms(user_id: int, phone: str, message: str):
    phone = phone or "unknown"
    with get_db() as conn:
        conn.execute(
            "INSERT INTO sms_log (user_id, phone, message) VALUES (?,?,?)",
            (user_id, phone, message)
        )


# ── Auth ──────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    sess = read_session(request)
    if sess:
        if sess.get("role") == "patient":
            return RedirectResponse("/patient/dashboard")
        return RedirectResponse("/doctor/dashboard")
    return _render("login.html", active_tab="login", error="")


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    sess = read_session(request)
    if sess:
        return RedirectResponse("/patient/dashboard" if sess.get("role") == "patient" else "/doctor/dashboard")
    return _render("login.html", active_tab="login", error="")


@app.post("/login")
async def do_login(request: Request, email: str = Form(...), password: str = Form(...)):
    user = get_user_by_email(email.strip().lower())
    if not user or not verify_password(password, user["password_hash"]):
        return _render("login.html", active_tab="login", error="Incorrect email or password.")
    resp = RedirectResponse(
        "/patient/dashboard" if user["role"] == "patient" else "/doctor/dashboard",
        status_code=303,
    )
    set_session_cookie(resp, user["id"], user["role"])
    return resp


@app.post("/register")
async def do_register(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    phone: str = Form(""),
    password: str = Form(...),
    role: str = Form("patient"),
):
    email = email.strip().lower()
    if get_user_by_email(email):
        return _render("login.html", active_tab="register", error="An account with that email already exists.")
    if role not in ("patient", "doctor"):
        role = "patient"
    pw_hash = hash_password(password)
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO users (name, email, phone, password_hash, role) VALUES (?,?,?,?,?)",
            (name.strip(), email, phone.strip() or None, pw_hash, role)
        )
        user_id = cur.lastrowid
    resp = RedirectResponse(
        "/patient/dashboard" if role == "patient" else "/doctor/dashboard",
        status_code=303,
    )
    set_session_cookie(resp, user_id, role)
    return resp


@app.get("/logout")
async def logout():
    resp = RedirectResponse("/", status_code=303)
    clear_session_cookie(resp)
    return resp


# ── Patient: Dashboard ────────────────────────────────────────────────────────

@app.get("/patient/dashboard", response_class=HTMLResponse)
async def patient_dashboard(request: Request):
    user = _patient_required(request)
    if not user:
        return RedirectResponse("/login")

    today = _today()
    week_start = _week_start()
    meds = get_active_medications(user["id"])
    today_logs = get_today_logs(user["id"], today)
    week_taken, week_total = get_weekly_stats(user["id"], week_start)
    latest_assessment = get_latest_assessment(user["id"])
    today_lifestyle = get_lifestyle_log(user["id"], today)
    streak = _calc_streak(user["id"])

    return _render_patient(
        "patient/dashboard.html", user, request,
        active="dashboard",
        meds=meds,
        today_logs_by_med=today_logs,
        week_taken=week_taken,
        week_total=week_total,
        latest_assessment=latest_assessment,
        today_lifestyle=today_lifestyle,
        streak=streak,
        articles=ARTICLES[:3],
    )


# ── Patient: Medications ──────────────────────────────────────────────────────

@app.get("/patient/medications", response_class=HTMLResponse)
async def patient_medications(request: Request):
    user = _patient_required(request)
    if not user:
        return RedirectResponse("/login")
    meds = get_active_medications(user["id"])
    return _render_patient("patient/medications.html", user, request, active="medications", meds=meds)


@app.post("/patient/medications")
async def add_medication(
    request: Request,
    name: str = Form(...),
    drug_class: str = Form("None"),
    dose_mg: str = Form(""),
    times_per_day: int = Form(1),
    reminder_times: str = Form(""),
):
    user = _patient_required(request)
    if not user:
        return RedirectResponse("/login")
    with get_db() as conn:
        conn.execute(
            "INSERT INTO medications (user_id, name, drug_class, dose_mg, times_per_day, reminder_times) VALUES (?,?,?,?,?,?)",
            (user["id"], name.strip(), drug_class, float(dose_mg) if dose_mg else None,
             times_per_day, reminder_times.strip() or None)
        )
    return _redirect("/patient/medications", "Medication added.")


@app.post("/patient/medications/{med_id}/delete")
async def delete_medication(med_id: int, request: Request):
    user = _patient_required(request)
    if not user:
        return RedirectResponse("/login")
    with get_db() as conn:
        conn.execute("UPDATE medications SET active=0 WHERE id=? AND user_id=?", (med_id, user["id"]))
    return _redirect("/patient/medications", "Medication removed.")


# ── Patient: Dose Log ─────────────────────────────────────────────────────────

@app.get("/patient/log", response_class=HTMLResponse)
async def patient_log(request: Request):
    user = _patient_required(request)
    if not user:
        return RedirectResponse("/login")
    today = _today()
    meds = get_active_medications(user["id"])
    today_logs = get_today_logs(user["id"], today)
    already_logged = bool(today_logs)

    # Last 7 days history with med names
    week_start = str(date.today() - timedelta(days=6))
    with get_db() as conn:
        rows = conn.execute("""
            SELECT d.scheduled_date, d.status, m.name AS med_name
            FROM dose_logs d JOIN medications m ON d.medication_id=m.id
            WHERE d.user_id=? AND d.scheduled_date>=?
            ORDER BY d.scheduled_date DESC
        """, (user["id"], week_start)).fetchall()
    history = [dict(r) for r in rows]

    return _render_patient(
        "patient/log.html", user, request,
        active="log",
        meds=meds,
        today_logs=today_logs,
        already_logged=already_logged,
        history=history,
    )


@app.post("/patient/log")
async def save_dose_log(request: Request):
    user = _patient_required(request)
    if not user:
        return RedirectResponse("/login")
    form = await request.form()
    today = _today()
    with get_db() as conn:
        for key, value in form.items():
            if not key.startswith("med_"):
                continue
            med_id = int(key[4:])
            if value not in ("taken", "missed"):
                continue
            conn.execute("""
                INSERT INTO dose_logs (user_id, medication_id, scheduled_date, status)
                VALUES (?,?,?,?)
                ON CONFLICT(medication_id, scheduled_date) DO UPDATE SET status=excluded.status
            """, (user["id"], med_id, today, value))
    return _redirect("/patient/log", "Dose log saved.")


# ── Patient: Assessment ───────────────────────────────────────────────────────

@app.get("/patient/assessment", response_class=HTMLResponse)
async def patient_assessment(request: Request):
    user = _patient_required(request)
    if not user:
        return RedirectResponse("/login")
    paper_mode = request.query_params.get("paper") == "1"
    meds = get_active_medications(user["id"])
    # Pre-fill drug classes from saved medications (up to 3)
    drug_classes = [m["drug_class"] for m in meds if m["drug_class"] != "None"][:3]
    total_pills = sum(m["times_per_day"] for m in meds)
    prefill = {
        "drug1_class": drug_classes[0] if len(drug_classes) > 0 else "Metformin",
        "drug2_class": drug_classes[1] if len(drug_classes) > 1 else "None",
        "drug3_class": drug_classes[2] if len(drug_classes) > 2 else "None",
        "total_pills": str(total_pills) if total_pills else "",
        "total_daily_frequency": str(total_pills) if total_pills else "",
        "self_glucose_monitoring": "0",
    }
    return _render_patient(
        "patient/assessment.html", user, request,
        active="assessment",
        result=None,
        prefill=prefill,
        paper_mode=paper_mode,
    )


@app.post("/patient/assessment", response_class=HTMLResponse)
async def run_patient_assessment(
    request: Request,
    # Section 1 — About You
    age: str = Form(...),
    sex: str = Form(...),
    weight_kg: str = Form(""),
    height_cm: str = Form(""),
    education: str = Form(...),
    marital_status: str = Form("married"),
    # Section 2 — Living Situation
    family_size: str = Form(...),
    residence: str = Form("urban"),
    distance_km: str = Form(...),
    # Section 3 — Finances
    income_score: str = Form(...),
    drug_cost_score: str = Form(...),
    consult_over_250: str = Form("0"),
    # Section 4 — Diabetes
    duration_months: str = Form(...),
    occupation: str = Form("employed"),
    has_htn: str = Form("0"),
    # comorbidities is a multi-value checkbox, handled via request.form()
    # Section 5 — Medications
    drug1_class: str = Form("Metformin"),
    drug2_class: str = Form("None"),
    drug3_class: str = Form("None"),
    total_pills: str = Form(...),
    total_daily_frequency: str = Form(...),
    new_drug_added: str = Form("0"),
    new_drug_available: str = Form("0"),
    # Section 6 — Labs
    fbs: str = Form(""),
    hba1c: str = Form(""),
    rbs: str = Form(""),
    time_since_lab_months: str = Form(""),
    self_glucose_monitoring: str = Form("0"),
    # Section 7 — Lifestyle & Support
    on_diabetic_diet: str = Form("0"),
    exercises: str = Form("0"),
    sleeps_adequately: str = Form("0"),
    smoking_alcohol: str = Form("0"),
    counselling_doctor: str = Form("0"),
    counselling_others: str = Form("0"),
    # Informational only — not passed to the model
    side_effects: str = Form(""),
):
    user = _patient_required(request)
    if not user:
        return RedirectResponse("/login")

    # Count comorbidities from checkbox list
    form = await request.form()
    comorbidities = form.getlist("comorbidities")
    comorbidity_count = len([c for c in comorbidities if c != "None"])
    # has_htn is asked separately but add it to count if also ticked in comorbidities
    # (has_htn field already captured above)

    form_data = {
        "age": age,
        "sex": sex,                          # already "0" or "1"
        "education": education,              # already "1"–"5"
        "occupation": occupation,            # "employed"/"retired"/"unemployed"
        "residence": residence,              # "urban"/"rural"/"tribal"
        "marital_status": marital_status,    # "married"/"single"/"divorced"/"widowed"
        "family_size": family_size,
        "distance_km": distance_km,
        "weight_kg": weight_kg,
        "height_cm": height_cm,
        "duration_months": duration_months,
        "time_since_lab_months": time_since_lab_months,
        "hba1c": hba1c,
        "fbs": fbs,
        "rbs": rbs,
        "drug1_class": drug1_class,
        "drug2_class": drug2_class,
        "drug3_class": drug3_class,
        "total_pills": total_pills,
        "total_daily_frequency": total_daily_frequency,
        "new_drug_added": new_drug_added,
        "new_drug_available": new_drug_available,
        "income_score": income_score,
        "drug_cost_score": drug_cost_score,
        "consult_over_250": consult_over_250,
        "counselling_others": counselling_others,
        "counselling_doctor": counselling_doctor,
        "on_diabetic_diet": on_diabetic_diet,
        "exercises": exercises,
        "sleeps_adequately": sleeps_adequately,
        "smoking_alcohol": smoking_alcohol,
        "self_glucose_monitoring": self_glucose_monitoring,
        "has_htn": has_htn,
        "comorbidity_count": str(comorbidity_count),
    }

    pred = predict(form_data)
    p_nonadherent = float(pred.get("p_nonadherent", 0))   # already 0–100

    # 3-tier risk: model output range is ~35–65% due to training distribution
    if p_nonadherent > 55.0:
        risk_tier = "HIGH"
        is_nonadherent = 1
    elif p_nonadherent >= 47.0:
        risk_tier = "MODERATE"
        is_nonadherent = 0
    else:
        risk_tier = "LOW"
        is_nonadherent = 0

    factors = [
        {
            "label": f["label"],
            "impact": "high" if f["importance"] > 0.05 else "medium",
            "feature": f["feature"],
            "category": _factor_category(f["feature"]),
            "reason": _factor_reason(f, form_data),
        }
        for f in pred.get("top_factors", [])
    ]

    category_breakdown = _categorize_factors(factors)
    dominant_category = max(category_breakdown, key=lambda c: len(category_breakdown[c])) if category_breakdown else None

    result = {
        "p_nonadherent": round(p_nonadherent, 1),
        "is_nonadherent": is_nonadherent,
        "risk_tier": risk_tier,
        "verdict": _verdict(p_nonadherent),
        "factors": factors,
        "category_breakdown": category_breakdown,
        "dominant_category": dominant_category,
    }

    with get_db() as conn:
        conn.execute(
            "INSERT INTO assessments (user_id, p_nonadherent, is_nonadherent, result_json) VALUES (?,?,?,?)",
            (user["id"], p_nonadherent, is_nonadherent, json.dumps({
                **pred, "p_nonadherent": p_nonadherent, "is_nonadherent": is_nonadherent,
            }))
        )

    # Re-populate prefill so the form retains entered values
    prefill = {
        "age": age, "sex": sex, "weight_kg": weight_kg, "height_cm": height_cm,
        "education": education, "marital_status": marital_status,
        "family_size": family_size, "residence": residence, "distance_km": distance_km,
        "income_score": income_score, "drug_cost_score": drug_cost_score,
        "consult_over_250": consult_over_250,
        "duration_months": duration_months, "occupation": occupation, "has_htn": has_htn,
        "comorbidities": comorbidities,
        "drug1_class": drug1_class, "drug2_class": drug2_class, "drug3_class": drug3_class,
        "total_pills": total_pills, "total_daily_frequency": total_daily_frequency,
        "new_drug_added": new_drug_added, "new_drug_available": new_drug_available,
        "fbs": fbs, "hba1c": hba1c, "rbs": rbs,
        "time_since_lab_months": time_since_lab_months,
        "self_glucose_monitoring": self_glucose_monitoring,
        "on_diabetic_diet": on_diabetic_diet, "exercises": exercises,
        "sleeps_adequately": sleeps_adequately, "smoking_alcohol": smoking_alcohol,
        "counselling_doctor": counselling_doctor,
        "side_effects": side_effects,
    }

    return _render_patient(
        "patient/assessment.html", user, request,
        active="assessment",
        result=result,
        prefill=prefill,
    )


# ── Patient: Education ────────────────────────────────────────────────────────

@app.get("/patient/education", response_class=HTMLResponse)
async def patient_education(request: Request, cat: str = ""):
    user = _patient_required(request)
    if not user:
        return RedirectResponse("/login")
    articles = get_by_category(cat) if cat else ARTICLES
    return _render_patient(
        "patient/education.html", user, request,
        active="education",
        articles=articles,
        active_cat=cat,
    )


@app.get("/patient/education/{slug}", response_class=HTMLResponse)
async def patient_article(slug: str, request: Request):
    user = _patient_required(request)
    if not user:
        return RedirectResponse("/login")
    article = get_article(slug)
    if not article:
        return RedirectResponse("/patient/education")
    related = [a for a in ARTICLES if a["category"] == article["category"] and a["slug"] != slug][:2]
    return _render_patient(
        "patient/education_article.html", user, request,
        active="education",
        article=article,
        related=related,
    )


# ── Patient: Lifestyle ────────────────────────────────────────────────────────

@app.get("/patient/lifestyle", response_class=HTMLResponse)
async def patient_lifestyle(request: Request):
    user = _patient_required(request)
    if not user:
        return RedirectResponse("/login")
    today = _today()
    today_log = get_lifestyle_log(user["id"], today)
    history = get_lifestyle_history(user["id"], days=7)
    return _render_patient(
        "patient/lifestyle.html", user, request,
        active="lifestyle",
        today_log=today_log,
        history=history,
    )


@app.post("/patient/lifestyle")
async def save_lifestyle(
    request: Request,
    on_diet: str = Form("0"),
    exercised: str = Form("0"),
    slept_adequately: str = Form("0"),
    glucose_reading: str = Form(""),
    notes: str = Form(""),
):
    user = _patient_required(request)
    if not user:
        return RedirectResponse("/login")
    today = _today()
    with get_db() as conn:
        conn.execute("""
            INSERT INTO lifestyle_logs (user_id, date, on_diet, exercised, slept_adequately, glucose_reading, notes)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(user_id, date) DO UPDATE SET
                on_diet=excluded.on_diet,
                exercised=excluded.exercised,
                slept_adequately=excluded.slept_adequately,
                glucose_reading=excluded.glucose_reading,
                notes=excluded.notes
        """, (
            user["id"], today,
            int(on_diet), int(exercised), int(slept_adequately),
            float(glucose_reading) if glucose_reading else None,
            notes.strip() or None,
        ))
    return _redirect("/patient/lifestyle", "Lifestyle log saved.")


# ── Patient: Notifications ────────────────────────────────────────────────────

@app.get("/patient/notifications", response_class=HTMLResponse)
async def patient_notifications(request: Request):
    user = _patient_required(request)
    if not user:
        return RedirectResponse("/login")
    with get_db() as conn:
        push_sub = conn.execute(
            "SELECT id FROM push_subscriptions WHERE user_id=? LIMIT 1", (user["id"],)
        ).fetchone()
    push_enabled = push_sub is not None
    sms_log = get_sms_log(user["id"])
    return _render_patient(
        "patient/notifications.html", user, request,
        active="notifications",
        push_enabled=push_enabled,
        sms_log=sms_log,
        vapid_public_key=VAPID_PUBLIC_KEY,
    )


@app.post("/notifications/subscribe")
async def subscribe_push(request: Request):
    user = _patient_required(request)
    if not user:
        return {"error": "Unauthorized"}, 401
    body = await request.json()
    endpoint = body.get("endpoint", "")
    keys = body.get("keys", {})
    p256dh = keys.get("p256dh", "")
    auth = keys.get("auth", "")
    with get_db() as conn:
        conn.execute("DELETE FROM push_subscriptions WHERE user_id=?", (user["id"],))
        conn.execute(
            "INSERT INTO push_subscriptions (user_id, endpoint, p256dh_key, auth_key) VALUES (?,?,?,?)",
            (user["id"], endpoint, p256dh, auth)
        )
    return {"ok": True}


@app.post("/notifications/unsubscribe")
async def unsubscribe_push(request: Request):
    user = _patient_required(request)
    if not user:
        return {"error": "Unauthorized"}
    with get_db() as conn:
        conn.execute("DELETE FROM push_subscriptions WHERE user_id=?", (user["id"],))
    return {"ok": True}


# ── Doctor: Dashboard ─────────────────────────────────────────────────────────

@app.get("/doctor/dashboard", response_class=HTMLResponse)
async def doctor_dashboard(request: Request):
    user = _doctor_required(request)
    if not user:
        return RedirectResponse("/login")

    patients_raw = get_all_patients()
    today = _today()
    week_start = _week_start()

    patients = []
    for p in patients_raw:
        assessment = get_latest_assessment(p["id"])
        week_taken, week_total = get_weekly_stats(p["id"], week_start)
        week_pct = (week_taken / week_total * 100) if week_total else 0
        p_score = assessment["p_nonadherent"] if assessment else None
        patients.append({
            **p,
            "p_nonadherent": p_score,
            "risk_tier": _risk_tier(p_score),
            "is_nonadherent": bool(p_score and p_score > 55.0),
            "last_assessment_date": assessment["created_at"] if assessment else None,
            "week_taken": week_taken,
            "week_total": week_total,
            "week_pct": week_pct,
        })

    high_risk = sum(1 for p in patients if p.get("risk_tier") == "HIGH")
    with get_db() as conn:
        assessed_today = conn.execute(
            "SELECT COUNT(DISTINCT user_id) FROM assessments WHERE DATE(created_at)=?", (today,)
        ).fetchone()[0]
        sms_sent = conn.execute("SELECT COUNT(*) FROM sms_log").fetchone()[0]

    stats = {
        "total": len(patients),
        "high_risk": high_risk,
        "assessed_today": assessed_today,
        "sms_sent": sms_sent,
    }

    return _render_doctor(
        "doctor/dashboard.html", user, request,
        active="dashboard",
        patients=patients,
        stats=stats,
    )


@app.get("/doctor/patient/{patient_id}", response_class=HTMLResponse)
async def doctor_patient_detail(patient_id: int, request: Request):
    user = _doctor_required(request)
    if not user:
        return RedirectResponse("/login")

    patient_user = get_user(patient_id)
    if not patient_user or patient_user["role"] != "patient":
        return RedirectResponse("/doctor/dashboard")

    latest = get_latest_assessment(patient_id)
    assessments = get_assessments(patient_id, limit=10)

    # Pull drug_class from latest result_json if available
    for a in assessments:
        try:
            rj = json.loads(a.get("result_json") or "{}")
            a["drug_class"] = rj.get("drug_class", "—")
        except Exception:
            a["drug_class"] = "—"

    week_start = _week_start()
    week_taken, week_total = get_weekly_stats(patient_id, week_start)
    week_pct = (week_taken / week_total * 100) if week_total else 0

    # Dose grid (last 30 dose log entries)
    thirty_ago = str(date.today() - timedelta(days=29))
    dose_logs = get_dose_grid(patient_id, thirty_ago, _today())
    dose_grid = [d["status"] for d in dose_logs]

    lifestyle = get_lifestyle_range(patient_id, week_start)
    sms_log = get_sms_log(patient_id)

    p_score = latest["p_nonadherent"] if latest else None
    patient_info = {
        **patient_user,
        "p_nonadherent": p_score,
        "risk_tier": _risk_tier(p_score),
        "is_nonadherent": bool(p_score and p_score > 55.0),
    }

    return _render_doctor(
        "doctor/patient_detail.html", user, request,
        active="dashboard",
        patient=patient_info,
        assessments=assessments,
        week_taken=week_taken,
        week_total=week_total,
        week_pct=week_pct,
        dose_grid=dose_grid,
        lifestyle=lifestyle,
        sms_log=sms_log,
    )


@app.post("/doctor/send-reminders")
async def send_bulk_reminders(request: Request):
    user = _doctor_required(request)
    if not user:
        return RedirectResponse("/login")

    patients = get_all_patients()
    count = 0
    for p in patients:
        assessment = get_latest_assessment(p["id"])
        if assessment and assessment["is_nonadherent"] and p.get("phone"):
            msg = (
                f"DA-AIIMS Reminder: Hi {p['name']}, your adherence risk score is elevated. "
                "Please take your medications as prescribed and consult your doctor if you have concerns."
            )
            _simulate_sms(p["id"], p["phone"], msg)
            count += 1

    return _redirect("/doctor/dashboard", f"Sent {count} reminder(s) to high-risk patients.")


@app.post("/doctor/send-reminder/{patient_id}")
async def send_single_reminder(patient_id: int, request: Request):
    user = _doctor_required(request)
    if not user:
        return RedirectResponse("/login")

    patient = get_user(patient_id)
    if patient and patient.get("phone"):
        msg = (
            f"DA-AIIMS: Hi {patient['name']}, your doctor has sent you a reminder to take your "
            "diabetes medications as prescribed. Please contact the clinic if you have any questions."
        )
        _simulate_sms(patient_id, patient["phone"], msg)
        return _redirect(f"/doctor/patient/{patient_id}", "Reminder sent.")

    return _redirect(f"/doctor/patient/{patient_id}", "No phone number on file for this patient.", "error")


# ── Legacy clinician tool (kept at /tool) ─────────────────────────────────────

@app.get("/tool", response_class=HTMLResponse)
async def tool_index(request: Request):
    sess = read_session(request)
    user = get_user(sess["id"]) if sess else None
    return _render("index.html", user=user)


@app.post("/tool/predict")
async def tool_predict(
    age: str = Form(...),
    sex: str = Form(...),
    education: str = Form(...),
    occupation: str = Form(...),
    residence: str = Form(...),
    marital_status: str = Form(...),
    family_size: str = Form(...),
    distance_km: str = Form(...),
    weight_kg: str = Form(""),
    height_cm: str = Form(""),
    duration_months: str = Form(...),
    time_since_lab_months: str = Form(""),
    hba1c: str = Form(""),
    fbs: str = Form(""),
    rbs: str = Form(""),
    drug1_class: str = Form(...),
    drug2_class: str = Form(...),
    drug3_class: str = Form(...),
    total_pills: str = Form("0"),
    total_daily_frequency: str = Form("0"),
    new_drug_added: str = Form("0"),
    new_drug_available: str = Form("0"),
    income_score: str = Form(...),
    drug_cost_score: str = Form(...),
    consult_over_250: str = Form("0"),
    counselling_others: str = Form("0"),
    counselling_doctor: str = Form("0"),
    on_diabetic_diet: str = Form("0"),
    exercises: str = Form("0"),
    sleeps_adequately: str = Form("0"),
    smoking_alcohol: str = Form("0"),
    self_glucose_monitoring: str = Form("0"),
    has_htn: str = Form("0"),
    comorbidity_count: str = Form("0"),
):
    form_data = {
        "age": age, "sex": sex, "education": education,
        "occupation": occupation, "residence": residence,
        "marital_status": marital_status, "family_size": family_size,
        "distance_km": distance_km, "weight_kg": weight_kg,
        "height_cm": height_cm, "duration_months": duration_months,
        "time_since_lab_months": time_since_lab_months,
        "hba1c": hba1c, "fbs": fbs, "rbs": rbs,
        "drug1_class": drug1_class, "drug2_class": drug2_class, "drug3_class": drug3_class,
        "total_pills": total_pills, "total_daily_frequency": total_daily_frequency,
        "new_drug_added": new_drug_added, "new_drug_available": new_drug_available,
        "income_score": income_score, "drug_cost_score": drug_cost_score,
        "consult_over_250": consult_over_250,
        "counselling_others": counselling_others, "counselling_doctor": counselling_doctor,
        "on_diabetic_diet": on_diabetic_diet, "exercises": exercises,
        "sleeps_adequately": sleeps_adequately, "smoking_alcohol": smoking_alcohol,
        "self_glucose_monitoring": self_glucose_monitoring,
        "has_htn": has_htn, "comorbidity_count": comorbidity_count,
    }
    result = predict(form_data)
    result_json = json.dumps(result)
    return _render("result.html", result=result, result_json=result_json)


# Keep old /predict for any existing bookmarks
@app.post("/predict")
async def old_predict(request: Request):
    return await tool_predict(**{k: v async for k, v in request.form()})
