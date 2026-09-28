from sqlalchemy import Column, Integer, String, Float, Boolean, JSON
from database import Base
from pydantic import BaseModel,Field


from typing import List, Optional

# ==========================================
# 1. SQLALCHEMY MODEL (The SQLite Database Table)
# ==========================================
class PatientDB(Base):
    __tablename__ = "patients"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=True)
    age = Column(Integer, nullable=True)
    weight_kg = Column(Float, nullable=True)
    height_cm = Column(Float, nullable=True)
    room = Column(String, nullable=True)
    risk = Column(Float, default=5.0)
    diagnosis = Column(String, nullable=True)
    assigned_nurse = Column(String, nullable=True)

    # --- Static Clinical Data ---
    baseline_risk = Column(Float, default=5.0)
    parkinsons = Column(Boolean, default=False)
    muscle_issues = Column(Boolean, default=False)
    vision_issues = Column(Boolean, default=False)
    recent_surgery = Column(String, default="None")
    
    # --- Hardware Link ---
    assigned_band = Column(String, nullable=True)
    assigned_spo2 = Column(String, nullable=True)

    # --- Live Data stored as JSON ---
    medications = Column(JSON, default=list)
    vitals = Column(JSON, default=dict)
    band_data = Column(JSON, default=dict)
    trend = Column(JSON, default=list)
    shap = Column(JSON, default=list)
    fall_history = Column(Integer, default=0)
    alert_age = Column(String, nullable=True)

# ==========================================
# 2. PYDANTIC MODELS (FastAPI Data Validation)
# ==========================================

# Schema for Hospital Receptionist
class PatientRegister(BaseModel):
    name: str
    age: int
    weight_kg: float
    height_cm: float
    room: str
    diagnosis: str

# Schema for the Nurse Intake Form
class PatientAssign(BaseModel):
    patient_id: str
    assigned_nurse: str
    ward_type: str = "general"
    
    # Neurological / Physical
    parkinsons: bool = False
    post_stroke: bool = False
    muscle_issues: bool = False
    vision_issues: bool = False
    recent_surgery: str = "None"
    
    # Pharmacy (FRIDs)
    on_opioids: bool = False
    on_antipsychotics: bool = False
    on_antidepressants: bool = False
    on_benzodiazepines: bool = False
    on_antiparkinson: bool = False
    on_antiepileptics: bool = False
    on_antihypertensives: bool = False
    on_diuretics: bool = False
    on_beta_blockers: bool = False
    on_alpha_blockers: bool = False
    on_nsaids: bool = False
    on_muscle_relaxants: bool = False
    
    # Devices
    assigned_band: str
    assigned_spo2: str

# Schema for MAX30100 ESP8266 Live Stream
class SpO2Reading(BaseModel):
    device_id: str
    heart_rate: float
    spo2: float

class Vitals(BaseModel):
    bp: str
    hr: int
    spo2: int
    rr: int
    temp: float

class BandData(BaseModel):
    steps: int
    motion_events: int
    gait_score: float

class ShapFeature(BaseModel):
    feature: str
    value: float
    positive: bool  

# Schema for Real-Time ESP8266 Data
class BandReading(BaseModel):
    patient_id: str
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    grav_x: float
    grav_y: float
    grav_z: float
    spo2: Optional[float] = None
    heart_rate: Optional[float] = None

class Patient(BaseModel):
    id: str
    name: Optional[str] = None
    age: Optional[int] = None
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None
    room: Optional[str] = None
    risk: float = 5.0
    diagnosis: Optional[str] = None
    assigned_nurse: Optional[str] = None
    baseline_risk: float = 5.0
    parkinsons: bool = False
    muscle_issues: bool = False
    vision_issues: bool = False
    recent_surgery: str = "None"
    assigned_band: Optional[str] = None
    assigned_spo2: Optional[str] = None
    medications: list = []
    vitals: dict = {}
    band_data: dict = {}
    trend: list = []
    shap: list = []
    fall_history: int = 0
    alert_age: Optional[str] = None

    model_config = {"from_attributes": True}