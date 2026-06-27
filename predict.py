from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODELS_DIR = Path(__file__).parent / "models"

FEATURE_NAMES = [
    "Age", "Sex", "weight_kg", "Height_cm", "BMI_kg_m2",
    "Education", "Distance_nearby_facility_from_house_approx_km",
    "Duration_of_illness_months", "Any_new_drug_added_in_the_last_visit",
    "Is_the_new_drug_easily_available", "Time_since_last_lab_investigation_months",
    "Last_FBS_value", "Last_HBAIC_value", "Last_RBS_value",
    "On_diabetic_diet", "Do_you_exercise", "Do_you_sleep_adequately",
    "Currently_any_smoking_alcohol_intake", "Duration_Years", "Family_Size",
    "Total_Pills", "Self_Glucose_Monitoring_Monthly",
    "Last_HBAIC_value_Missing_Flag", "Last_FBS_value_Missing_Flag",
    "Last_RBS_value_Missing_Flag", "Time_since_last_lab_investigation_months_Missing_Flag",
    "BMI_kg_m2_Missing_Flag", "weight_kg_Missing_Flag", "Height_cm_Missing_Flag",
    "Total_Drugs_Prescribed", "Total_Daily_Frequency",
    "Income_Score", "Drug_Cost_Score", "Financial_Stress_Ratio",
    "Comorbidity_Count", "Has_HTN",
    "Occupation_Retired", "Occupation_Unemployed",
    "Residence_tribal", "Residence_urban",
    "Marital_status_Single", "Marital_status_divorced",
    "Do_you_get_counselling_regarding_disease_drug_use_drug_adherance_others",
    "Do_you_get_counselling_regarding_disease_drug_use_drug_adherance_treating_doctor",
    "Rough_monthly_expenses_on_consultation_rupees_250",
    "Drug_1_Group_Metformin", "Drug_1_Group_New_Gen_Oral", "Drug_1_Group_None",
    "Drug_1_Group_Other_Combination", "Drug_1_Group_Sulfonylurea",
    "Drug_2_Group_Metformin", "Drug_2_Group_New_Gen_Oral", "Drug_2_Group_None",
    "Drug_2_Group_Other_Combination", "Drug_2_Group_Sulfonylurea",
    "Drug_3_Group_Metformin", "Drug_3_Group_New_Gen_Oral", "Drug_3_Group_None",
    "Drug_3_Group_Other_Combination", "Drug_3_Group_Sulfonylurea",
]

FEATURE_LABELS = {
    "Last_RBS_value": "Random Blood Sugar",
    "Last_HBAIC_value": "HbA1c Level",
    "Last_FBS_value": "Fasting Blood Sugar",
    "Financial_Stress_Ratio": "Financial Stress (Drug Cost/Income)",
    "Total_Daily_Frequency": "Total Daily Doses",
    "Total_Drugs_Prescribed": "Number of Drugs",
    "Duration_of_illness_months": "Duration of Illness",
    "Age": "Patient Age",
    "BMI_kg_m2": "BMI",
    "Distance_nearby_facility_from_house_approx_km": "Distance to Facility",
    "Do_you_sleep_adequately": "Adequate Sleep",
    "Do_you_exercise": "Regular Exercise",
    "On_diabetic_diet": "Following Diabetic Diet",
    "Currently_any_smoking_alcohol_intake": "Smoking / Alcohol Use",
    "Self_Glucose_Monitoring_Monthly": "Self Glucose Monitoring Frequency",
    "Income_Score": "Income Level",
    "Drug_Cost_Score": "Drug Cost Level",
    "Comorbidity_Count": "Number of Comorbidities",
    "Has_HTN": "Has Hypertension",
    "Family_Size": "Family Size",
    "Duration_Years": "Duration of Illness (yrs)",
    "weight_kg": "Weight (kg)",
    "Height_cm": "Height (cm)",
    "Total_Pills": "Total Pills Per Day",
    "Education": "Education Level",
    "Sex": "Sex",
    "Any_new_drug_added_in_the_last_visit": "New Drug Added Recently",
    "Do_you_get_counselling_regarding_disease_drug_use_drug_adherance_others": "Counselling (Others)",
    "Do_you_get_counselling_regarding_disease_drug_use_drug_adherance_treating_doctor": "Counselling (Treating Doctor)",
    "Rough_monthly_expenses_on_consultation_rupees_250": "Consultation Cost ≥ ₹250/month",
}

RECOMMENDATIONS = {
    "Last_RBS_value": {
        "icon": "🩸",
        "title": "Glycemic Control",
        "text": "Elevated random blood sugar is a key adherence risk factor. Review current regimen and consider dose adjustment or intensification.",
    },
    "Last_HBAIC_value": {
        "icon": "📊",
        "title": "Long-term Glucose Management",
        "text": "HbA1c levels indicate suboptimal long-term control. Reinforce the importance of consistent medication use and dietary adherence.",
    },
    "Last_FBS_value": {
        "icon": "🍽️",
        "title": "Fasting Glucose",
        "text": "High fasting glucose suggests morning dose adherence issues. Consider medication timing counselling.",
    },
    "Financial_Stress_Ratio": {
        "icon": "💊",
        "title": "Financial Barriers",
        "text": "High drug cost relative to income is a significant barrier. Explore generic alternatives, Jan Aushadhi scheme, or PMJAY drug benefits.",
    },
    "Total_Daily_Frequency": {
        "icon": "⏰",
        "title": "Regimen Complexity",
        "text": "High daily dosing frequency increases the chance of missed doses. Discuss fixed-dose combinations or once-daily formulations with the prescriber.",
    },
    "Total_Drugs_Prescribed": {
        "icon": "💊",
        "title": "Polypharmacy",
        "text": "Multiple medications increase pill burden. Review the regimen for any medications that can be consolidated.",
    },
    "Distance_nearby_facility_from_house_approx_km": {
        "icon": "🏥",
        "title": "Facility Access",
        "text": "Long distance to the healthcare facility may limit refill adherence. Explore telepharmacy, community health worker follow-up, or closer pharmacy tie-ups.",
    },
    "Do_you_sleep_adequately": {
        "icon": "😴",
        "title": "Sleep & Routine",
        "text": "Poor sleep disrupts medication routines. Counsel on sleep hygiene and linking medication timing to daily habits.",
    },
    "Do_you_exercise": {
        "icon": "🏃",
        "title": "Physical Activity",
        "text": "Regular exercise improves glycemic control and may reinforce healthy routines including medication adherence.",
    },
    "Currently_any_smoking_alcohol_intake": {
        "icon": "🚭",
        "title": "Substance Use",
        "text": "Smoking or alcohol use is associated with lower medication adherence. Provide brief motivational counselling and referral if needed.",
    },
    "Self_Glucose_Monitoring_Monthly": {
        "icon": "📱",
        "title": "Self-Monitoring",
        "text": "Infrequent self-glucose monitoring reduces awareness of glycemic status. Encourage regular home monitoring and logbook use.",
    },
}

DEFAULT_RECOMMENDATIONS = [
    {
        "icon": "📋",
        "title": "Regular Follow-up",
        "text": "Schedule quarterly reviews to reassess adherence barriers and medication effectiveness.",
    },
    {
        "icon": "🤝",
        "title": "Peer Support",
        "text": "Connecting patients with diabetes support groups can improve adherence through shared experiences.",
    },
]

_model = joblib.load(MODELS_DIR / "model.pkl")
_imputer = joblib.load(MODELS_DIR / "imputer.pkl")
_scaler = joblib.load(MODELS_DIR / "scaler.pkl")

# Extract base RF estimator for feature importances (CalibratedClassifierCV wraps it)
_base_estimator = (
    _model.calibrated_classifiers_[0].estimator
    if hasattr(_model, "calibrated_classifiers_")
    else _model
)
_feature_importances = np.array(_base_estimator.feature_importances_)


def _drug_onehot(drug_class: str, slot: int) -> dict:
    groups = ["Metformin", "New_Gen_Oral", "None", "Other_Combination", "Sulfonylurea"]
    prefix = f"Drug_{slot}_Group_"
    return {f"{prefix}{g}": int(drug_class == g) for g in groups}


def engineer_features(form: dict) -> pd.DataFrame:
    def floatval(key):
        v = form.get(key, "").strip()
        return (float(v), False) if v else (np.nan, True)

    age = float(form["age"])
    sex = int(form["sex"])
    education = int(form["education"])
    family_size = float(form["family_size"])
    distance = float(form["distance_km"])
    duration_months = float(form["duration_months"])
    occupation = form.get("occupation", "other")
    residence = form.get("residence", "rural")
    marital = form.get("marital_status", "married")

    weight, weight_miss = floatval("weight_kg")
    height, height_miss = floatval("height_cm")
    hba1c, hba1c_miss = floatval("hba1c")
    fbs, fbs_miss = floatval("fbs")
    rbs, rbs_miss = floatval("rbs")
    time_lab, time_lab_miss = floatval("time_since_lab_months")

    bmi = np.nan
    bmi_miss = True
    if not weight_miss and not height_miss and height > 0:
        bmi = weight / ((height / 100) ** 2)
        bmi_miss = False

    new_drug = int(form.get("new_drug_added", "0"))
    new_drug_available = int(form.get("new_drug_available", "0"))
    total_pills = float(form.get("total_pills", 0) or 0)
    total_freq = float(form.get("total_daily_frequency", 0) or 0)
    self_glucose = float(form.get("self_glucose_monitoring", 0) or 0)

    on_diet = int(form.get("on_diabetic_diet", "0"))
    exercises = int(form.get("exercises", "0"))
    sleeps = int(form.get("sleeps_adequately", "0"))
    smoking_alcohol = int(form.get("smoking_alcohol", "0"))

    income_score = float(form.get("income_score", 3))
    drug_cost_score = float(form.get("drug_cost_score", 3))
    financial_stress = drug_cost_score / income_score if income_score > 0 else 0.0
    consult_250 = int(form.get("consult_over_250", "0"))
    counselling_others = int(form.get("counselling_others", "0"))
    counselling_doctor = int(form.get("counselling_doctor", "0"))

    comorbidity_count = int(form.get("comorbidity_count", 0) or 0)
    has_htn = int(form.get("has_htn", "0"))

    drug1 = form.get("drug1_class", "None")
    drug2 = form.get("drug2_class", "None")
    drug3 = form.get("drug3_class", "None")
    total_drugs = sum(1 for d in [drug1, drug2, drug3] if d != "None")

    row = {
        "Age": age,
        "Sex": sex,
        "weight_kg": weight,
        "Height_cm": height,
        "BMI_kg_m2": bmi,
        "Education": education,
        "Distance_nearby_facility_from_house_approx_km": distance,
        "Duration_of_illness_months": duration_months,
        "Any_new_drug_added_in_the_last_visit": new_drug,
        "Is_the_new_drug_easily_available": new_drug_available,
        "Time_since_last_lab_investigation_months": time_lab,
        "Last_FBS_value": fbs,
        "Last_HBAIC_value": hba1c,
        "Last_RBS_value": rbs,
        "On_diabetic_diet": on_diet,
        "Do_you_exercise": exercises,
        "Do_you_sleep_adequately": sleeps,
        "Currently_any_smoking_alcohol_intake": smoking_alcohol,
        "Duration_Years": duration_months / 12,
        "Family_Size": family_size,
        "Total_Pills": total_pills,
        "Self_Glucose_Monitoring_Monthly": self_glucose,
        "Last_HBAIC_value_Missing_Flag": int(hba1c_miss),
        "Last_FBS_value_Missing_Flag": int(fbs_miss),
        "Last_RBS_value_Missing_Flag": int(rbs_miss),
        "Time_since_last_lab_investigation_months_Missing_Flag": int(time_lab_miss),
        "BMI_kg_m2_Missing_Flag": int(bmi_miss),
        "weight_kg_Missing_Flag": int(weight_miss),
        "Height_cm_Missing_Flag": int(height_miss),
        "Total_Drugs_Prescribed": total_drugs,
        "Total_Daily_Frequency": total_freq,
        "Income_Score": income_score,
        "Drug_Cost_Score": drug_cost_score,
        "Financial_Stress_Ratio": financial_stress,
        "Comorbidity_Count": comorbidity_count,
        "Has_HTN": has_htn,
        "Occupation_Retired": int(occupation == "retired"),
        "Occupation_Unemployed": int(occupation == "unemployed"),
        "Residence_tribal": int(residence == "tribal"),
        "Residence_urban": int(residence == "urban"),
        "Marital_status_Single": int(marital == "single"),
        "Marital_status_divorced": int(marital == "divorced"),
        "Do_you_get_counselling_regarding_disease_drug_use_drug_adherance_others": counselling_others,
        "Do_you_get_counselling_regarding_disease_drug_use_drug_adherance_treating_doctor": counselling_doctor,
        "Rough_monthly_expenses_on_consultation_rupees_250": consult_250,
    }
    row.update(_drug_onehot(drug1, 1))
    row.update(_drug_onehot(drug2, 2))
    row.update(_drug_onehot(drug3, 3))

    return pd.DataFrame([row], columns=FEATURE_NAMES)


def predict(form: dict) -> dict:
    df = engineer_features(form)
    X_imputed = pd.DataFrame(_imputer.transform(df), columns=FEATURE_NAMES)
    X_scaled = pd.DataFrame(_scaler.transform(X_imputed), columns=FEATURE_NAMES)

    proba = _model.predict_proba(X_scaled)[0]
    # classes_: [0=Non-Adherent, 1=Adherent] — proba[0]=P(Non-Adherent)
    p_nonadherent = float(proba[0])
    p_adherent = float(proba[1])
    is_nonadherent = p_nonadherent >= 0.5

    # Use model-level feature importances (global attribution from Random Forest)
    # Scaled patient values: positive = above training mean (scaled > 0), negative = below
    patient_scaled = X_scaled.values[0]
    top5_idx = np.argsort(_feature_importances)[::-1][:5].tolist()
    top_factors = [
        {
            "feature": FEATURE_NAMES[i],
            "label": FEATURE_LABELS.get(FEATURE_NAMES[i], FEATURE_NAMES[i].replace("_", " ")),
            "importance": float(_feature_importances[i]),
            # scaled_value > 0 means patient is above the training mean for that feature
            "patient_direction": float(patient_scaled[i]),
        }
        for i in top5_idx
    ]

    recs = []
    for f in top_factors:
        feat = f["feature"]
        if feat in RECOMMENDATIONS and len(recs) < 4:
            recs.append(RECOMMENDATIONS[feat])

    while len(recs) < 2:
        recs.append(DEFAULT_RECOMMENDATIONS[len(recs) % len(DEFAULT_RECOMMENDATIONS)])

    return {
        "is_nonadherent": is_nonadherent,
        "p_nonadherent": round(p_nonadherent * 100, 1),
        "p_adherent": round(p_adherent * 100, 1),
        "top_factors": top_factors,
        "recommendations": recs,
    }
