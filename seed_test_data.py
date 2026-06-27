"""
Seed script — inserts 3 test patients + 1 test doctor into the DB,
adds medications, and runs an assessment for each patient.

Run from the webapp/ directory:
    cd webapp && python seed_test_data.py

Expected assessment results:
    Rajesh Kumar   → HIGH RISK   (>55%)
    Priya Sharma   → LOW RISK    (<47%)
    Mohammed Salim → MODERATE    (47–55%)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from auth import hash_password
from db import get_db, init_db
from predict import predict

init_db()

# ── helpers ──────────────────────────────────────────────────────────────────

def _upsert_user(name, email, phone, password, role):
    with get_db() as conn:
        existing = conn.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()
        if existing:
            print(f"  already exists: {email}")
            return existing[0]
        cur = conn.execute(
            "INSERT INTO users (name, email, phone, password_hash, role) VALUES (?,?,?,?,?)",
            (name, email, phone, hash_password(password), role)
        )
        print(f"  created: {email}")
        return cur.lastrowid


def _add_med(user_id, name, drug_class, times_per_day):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO medications (user_id, name, drug_class, dose_mg, times_per_day) VALUES (?,?,?,?,?)",
            (user_id, name, drug_class, None, times_per_day)
        )


def _run_assessment(user_id, form_data):
    pred = predict(form_data)
    p = float(pred.get("p_nonadherent", 0))
    is_na = int(p > 55.0)
    with get_db() as conn:
        conn.execute(
            "INSERT INTO assessments (user_id, p_nonadherent, is_nonadherent, result_json) VALUES (?,?,?,?)",
            (user_id, p, is_na, json.dumps({**pred, "p_nonadherent": p, "is_nonadherent": is_na}))
        )
    tier = "HIGH" if p > 55 else ("MODERATE" if p >= 47 else "LOW")
    print(f"  assessment -> {p:.1f}% [{tier}]")


# ── Doctor ────────────────────────────────────────────────────────────────────

print("\n[Doctor]")
_upsert_user("Dr. Anitha Rao", "doctor@daims.test", "9900001111", "doctor123", "doctor")


# ── Patient 1 — HIGH RISK ─────────────────────────────────────────────────────

print("\n[Patient 1 — Rajesh Kumar — expected HIGH RISK]")
uid1 = _upsert_user("Rajesh Kumar", "rajesh@daims.test", "9900002222", "patient123", "patient")
_add_med(uid1, "Glipizide",  "Sulfonylurea", 2)
_add_med(uid1, "Metformin",  "Metformin",    2)

_run_assessment(uid1, {
    "age": "67", "sex": "1",
    "weight_kg": "88", "height_cm": "164",
    "education": "1", "marital_status": "single",
    "family_size": "2", "residence": "tribal", "distance_km": "18",
    "income_score": "2", "drug_cost_score": "5", "consult_over_250": "1",
    "duration_months": "216", "occupation": "retired",
    "has_htn": "1", "comorbidity_count": "2",
    "drug1_class": "Sulfonylurea", "drug2_class": "Metformin", "drug3_class": "None",
    "total_pills": "4", "total_daily_frequency": "3",
    "new_drug_added": "1", "new_drug_available": "0",
    "fbs": "185", "hba1c": "10.5", "rbs": "310",
    "time_since_lab_months": "8", "self_glucose_monitoring": "0",
    "on_diabetic_diet": "0", "exercises": "0", "sleeps_adequately": "0",
    "smoking_alcohol": "1", "counselling_doctor": "0", "counselling_others": "0",
})


# ── Patient 2 — LOW RISK ──────────────────────────────────────────────────────

print("\n[Patient 2 — Priya Sharma — expected LOW RISK]")
uid2 = _upsert_user("Priya Sharma", "priya@daims.test", "9900003333", "patient123", "patient")
_add_med(uid2, "Metformin 500mg", "Metformin", 2)

_run_assessment(uid2, {
    "age": "44", "sex": "0",
    "weight_kg": "62", "height_cm": "160",
    "education": "5", "marital_status": "married",
    "family_size": "4", "residence": "urban", "distance_km": "2",
    "income_score": "4", "drug_cost_score": "1", "consult_over_250": "0",
    "duration_months": "36", "occupation": "employed",
    "has_htn": "0", "comorbidity_count": "0",
    "drug1_class": "Metformin", "drug2_class": "None", "drug3_class": "None",
    "total_pills": "2", "total_daily_frequency": "2",
    "new_drug_added": "0", "new_drug_available": "0",
    "fbs": "105", "hba1c": "6.8", "rbs": "128",
    "time_since_lab_months": "2", "self_glucose_monitoring": "10",
    "on_diabetic_diet": "1", "exercises": "1", "sleeps_adequately": "1",
    "smoking_alcohol": "0", "counselling_doctor": "1", "counselling_others": "1",
})


# ── Patient 3 — MODERATE RISK ─────────────────────────────────────────────────

print("\n[Patient 3 — Mohammed Salim — expected MODERATE RISK]")
uid3 = _upsert_user("Mohammed Salim", "mohammed@daims.test", "9900004444", "patient123", "patient")
_add_med(uid3, "Jardiance", "New_Gen_Oral", 1)
_add_med(uid3, "Metformin", "Metformin",    2)

_run_assessment(uid3, {
    "age": "55", "sex": "1",
    "weight_kg": "78", "height_cm": "168",
    "education": "2", "marital_status": "married",
    "family_size": "3", "residence": "tribal", "distance_km": "12",
    "income_score": "2", "drug_cost_score": "4", "consult_over_250": "1",
    "duration_months": "96", "occupation": "employed",
    "has_htn": "1", "comorbidity_count": "2",
    "drug1_class": "New_Gen_Oral", "drug2_class": "Metformin", "drug3_class": "None",
    "total_pills": "3", "total_daily_frequency": "2",
    "new_drug_added": "1", "new_drug_available": "0",
    "fbs": "148", "hba1c": "8.4", "rbs": "240",
    "time_since_lab_months": "5", "self_glucose_monitoring": "2",
    "on_diabetic_diet": "0", "exercises": "0", "sleeps_adequately": "1",
    "smoking_alcohol": "0", "counselling_doctor": "1", "counselling_others": "0",
})


print("\nDone. Credentials:")
print("  Doctor:   doctor@daims.test  /  doctor123")
print("  Patient1: rajesh@daims.test  /  patient123  (HIGH RISK)")
print("  Patient2: priya@daims.test   /  patient123  (LOW RISK)")
print("  Patient3: mohammed@daims.test / patient123  (MODERATE RISK)")
