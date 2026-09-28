from sqlalchemy.orm import Session
from models import PatientDB, PatientRegister, PatientAssign
from datetime import datetime

def get_patient(db: Session, patient_id: str):
    return db.query(PatientDB).filter(PatientDB.id == patient_id).first()

def get_all_patients(db: Session):
    return db.query(PatientDB).all()

def register_patient(db: Session, reg_data: PatientRegister, patient_id: str):
    db_patient = PatientDB(id=patient_id)
    db.add(db_patient)
    
    db_patient.name = reg_data.name
    db_patient.age = reg_data.age
    db_patient.weight_kg = reg_data.weight_kg
    db_patient.height_cm = reg_data.height_cm
    db_patient.room = reg_data.room
    db_patient.diagnosis = reg_data.diagnosis
    
    db.commit()
    db.refresh(db_patient)
    return db_patient

def assign_patient_band(db: Session, assign_data: PatientAssign, baseline_risk: float):
    db_patient = get_patient(db, assign_data.patient_id)
    if not db_patient:
        return None
        
    db_patient.assigned_nurse = assign_data.assigned_nurse
    db_patient.parkinsons = assign_data.parkinsons
    db_patient.muscle_issues = assign_data.muscle_issues
    db_patient.vision_issues = assign_data.vision_issues
    db_patient.recent_surgery = assign_data.recent_surgery
    db_patient.assigned_band = assign_data.assigned_band
    db_patient.assigned_spo2 = assign_data.assigned_spo2
    db_patient.baseline_risk = baseline_risk
    db_patient.risk = baseline_risk

    # Gather medications into a list for the JSON column
    meds = []
    if assign_data.on_opioids: meds.append("on_opioids")
    if assign_data.on_antipsychotics: meds.append("on_antipsychotics")
    if assign_data.on_antidepressants: meds.append("on_antidepressants")
    if assign_data.on_benzodiazepines: meds.append("on_benzodiazepines")
    if assign_data.on_antiparkinson: meds.append("on_antiparkinson")
    if assign_data.on_antiepileptics: meds.append("on_antiepileptics")
    if assign_data.on_antihypertensives: meds.append("on_antihypertensives")
    if assign_data.on_diuretics: meds.append("on_diuretics")
    if assign_data.on_beta_blockers: meds.append("on_beta_blockers")
    if assign_data.on_alpha_blockers: meds.append("on_alpha_blockers")
    if assign_data.on_nsaids: meds.append("on_nsaids")
    if assign_data.on_muscle_relaxants: meds.append("on_muscle_relaxants")
    db_patient.medications = meds
    
    # Initialize JSON dictionary fields if they are empty
    if not db_patient.trend:
        db_patient.trend = [baseline_risk]
    if not db_patient.vitals:
        db_patient.vitals = {"bp": "120/80", "hr": 70, "spo2": 98, "rr": 16, "temp": 98.6}
    if not db_patient.band_data:
        db_patient.band_data = {"steps": 0, "motion_events": 0, "gait_score": 100, "frid_score": 0.0, "age_multiplier": 1.0}
    if not db_patient.shap:
        db_patient.shap = [{"feature": "Static Baseline", "value": baseline_risk, "positive": True}]
        
    db_patient.alert_age = datetime.now().strftime('%H:%M:%S')
    
    db.commit()
    db.refresh(db_patient)
    return db_patient

def update_db_vitals(db: Session, device_id: str, heart_rate: float, spo2: float):
    # Find active patient mapped to this SpO2 sensor
    db_patient = db.query(PatientDB).filter(PatientDB.assigned_spo2 == device_id).first()
    if db_patient:
        vitals = dict(db_patient.vitals) if db_patient.vitals else {}
        vitals["hr"] = heart_rate
        vitals["spo2"] = spo2
        db_patient.vitals = vitals
        db.commit()
        return db_patient.id
    return None

def update_live_risk(db: Session, patient_id: str, new_risk: float, alerts: list, band_updates: dict, vitals_updates: dict):
    db_patient = get_patient(db, patient_id)
    if not db_patient:
        return None
        
    db_patient.risk = new_risk
    db_patient.alert_age = datetime.now().strftime('%H:%M:%S')
    
    # 1. Update the 7-day trend array (keep max 10 points for the chart)
    current_trend = list(db_patient.trend) if db_patient.trend else []
    current_trend.append(new_risk)
    if len(current_trend) > 10:
        current_trend.pop(0)
    db_patient.trend = current_trend
    
    # 2. Update the "Reasoning" for the Dashboard UI
    reasons = [{"feature": "Static Baseline", "value": db_patient.baseline_risk, "positive": True}]
    for alert in alerts:
        reasons.append({"feature": alert, "value": round(new_risk - db_patient.baseline_risk, 1), "positive": True})
    db_patient.shap = reasons
    
    # 3. Safely update nested JSON dictionaries (optimized for throttling)
    current_band = db_patient.band_data.copy() if db_patient.band_data else {}
    current_band.update(band_updates)
    
    db_patient.band_data = current_band
    
    current_vitals = db_patient.vitals.copy() if db_patient.vitals else {}
    current_vitals.update(vitals_updates)
    db_patient.vitals = current_vitals
    
    db.commit()
    return db_patient