"""
ClinicalGuard Fall Prediction Algorithm
========================================
Complete medically-grounded fall risk engine.
Replaces the prototype logic in the original backend.

Data sources:
  - Morse Fall Scale (Morse et al. 1989; validated 206,846 pts, AUC 0.825)
  - FRID classification (BMC Geriatrics 2023; JAMDA 2024 longitudinal)
  - Biomechanical thresholds (derived from 2000-sample real IMU dataset)
  - Age/condition multipliers (WHO Falls Report; MFS age-stratified AUC data)
  - Fall vs intentional-bend discrimination (temporal profile analysis)
"""

import math
import time
import warnings
import os
warnings.filterwarnings("ignore", category=FutureWarning)

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
import google.generativeai as genai
from database import engine, get_db, create_tables
from models import PatientAssign, PatientRegister, SpO2Reading, BandReading, Patient
import crud

app = FastAPI()
gemini_api_key = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")
genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel('gemini-pro')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🚀 Hybrid Clinical-AI Backend is Running!")

async def get_ai_reasoning(patient, alerts):
    prompt = f"""
    Act as a clinical falls-prevention specialist. 
    Patient: {patient.name}, Age {patient.age}. 
    Medical History: Parkinson's={patient.parkinsons}, Recent Surgery={patient.recent_surgery}.
    Current Sensor Alerts: {', '.join(alerts)}.
    
    In 2 short sentences, explain the physiological risk and provide 1 immediate nursing intervention. 
    Be concise and professional.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return "AI reasoning currently unavailable. Monitor patient posture manually."

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: ENUMERATIONS — clinical states and patient categories
# ─────────────────────────────────────────────────────────────────────────────

class PosturalState(Enum):
    """IPSG.6-aligned three-state temporal classifier."""
    HOMEOSTASIS         = "homeostasis"           # score < 30, stable
    PRODROMAL           = "prodromal_instability"  # score 30–59, ≥3 windows
    ACTIVE_FALL         = "active_fall_trajectory" # score ≥ 60 or any critical


class FallCategory(Enum):
    """IPSG.6 fall type classification (Quigley framework)."""
    ACCIDENTAL          = "accidental"             # low-risk patient, environmental
    ANTICIPATED_PHYSIO  = "anticipated_physiological"  # identifiable risk factors
    UNANTICIPATED_PHYSIO = "unanticipated_physiological"  # seizure, syncope, arrhythmia
    NOT_A_FALL          = "not_a_fall"             # intentional bend, movement artifact


class MotionEvent(Enum):
    """Output of the fall vs. intentional-bend discriminator."""
    STABLE              = "stable"
    INTENTIONAL_BEND    = "intentional_bend"       # controlled descent, recovers
    POSTURAL_LEAN       = "postural_lean"          # slow lean, no impact
    PRE_FALL_WARNING    = "pre_fall_warning"       # prodromal instability
    FALL_TRAJECTORY     = "fall_trajectory"        # active, unrecovered tilt
    CONFIRMED_FALL      = "confirmed_fall"         # impact + no recovery
    SLOW_NEURO_FALL     = "slow_neuro_fall"        # PD-type: slow rate, no recovery


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: THRESHOLDS — all derived from real data or published literature
# ─────────────────────────────────────────────────────────────────────────────

class Thresholds:
    # ── BIOMECHANICAL (derived from 2000-sample real IMU CSV) ─────────────────
    # Optimal thresholds maximising sensitivity + specificity from your dataset
    TILT_WARN_DEG        = 28.7   # trunk tilt warning (data-derived, replaces estimated 20°)
    TILT_CRITICAL_DEG    = 38.0   # trunk tilt critical
    SMV_WARN             = 16.3   # signal magnitude vector warning (m/s²) — data-derived
    SMV_CRITICAL         = 28.0   # signal magnitude vector critical
    PITCH_WARN_DEG       = 15.2   # sagittal pitch warning
    ROLL_WARN_DEG        = 13.8   # lateral roll warning
    JERK_WARN            = 3.85   # resultant jerk warning (m/s³) — data-derived
    JERK_CRITICAL        = 7.0    # resultant jerk critical

    # ── FALL vs INTENTIONAL-BEND DISCRIMINATOR ────────────────────────────────
    # Based on temporal motion profile analysis
    BEND_MAX_TILT_RATE   = 15.0   # °/s — controlled descent stays below this
    FALL_MIN_TILT_RATE   = 30.0   # °/s — uncontrolled fall exceeds this
    BEND_MIN_TIME_TO_PEAK = 2.0   # seconds — slow deliberate bend takes >2 s
    FALL_MAX_TIME_TO_PEAK = 0.8   # seconds — fall reaches peak tilt in <0.8 s
    BEND_MAX_JERK        = 1.5    # m/s³ — smooth bend has low jerk
    RECOVERY_WINDOW_S    = 8.0    # seconds — must return toward baseline within this
    RECOVERY_THRESHOLD_DEG = 15.0 # must drop by at least this many degrees to count as recovery
    POST_IMPACT_SMV      = 16.3   # confirmed impact threshold
    SLOW_NEURO_TILT_RATE = 18.0   # °/s — PD slow fall starts here
    SDI_NEURO_THRESHOLD  = 1.0    # SDI peaks/s below which neuro discrimination applies

    # ── PHYSIOLOGICAL ─────────────────────────────────────────────────────────
    SPO2_WARN            = 94     # % — cerebral hypoperfusion risk begins
    SPO2_CRITICAL        = 90     # % — hypoxaemia, immediate syncope risk
    HR_DROP_WARN         = 20     # bpm sudden drop — vasovagal precursor
    BP_ORTHO_DROP        = 20     # mmHg systolic drop on standing — orthostatic hypotension

    # ── GAIT ─────────────────────────────────────────────────────────────────
    GAIT_SPEED_WARN      = 0.8    # m/s — below Studenski 2011 JAMA threshold
    GAIT_SPEED_CRITICAL  = 0.5    # m/s — severe reduction, Tinetti-correlated
    TUG_HIGH_RISK_S      = 12.0   # seconds — Podsiadlo & Richardson 1991 cutoff
    SDI_WARN             = 1.4    # peaks/s — elderly deterioration zone
    SDI_CRITICAL         = 1.0    # peaks/s — neurological deficit zone

    # ── MORSE FALL SCALE ─────────────────────────────────────────────────────
    MFS_MEDIUM_RISK      = 25     # score 25–44
    MFS_HIGH_RISK        = 45     # score ≥ 45
    MFS_OB_GYN_HIGH      = 40     # adjusted cutoff for obstetric patients (PLoS One 2024)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: PATIENT PROFILE — static clinical factors
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PatientProfile:
    patient_id: str
    name: str
    age: int
    weight_kg: float
    height_cm: float

    # ── Neurological conditions
    parkinsons: bool = False
    post_stroke: bool = False
    post_cranial_surgery: bool = False  # acute ≤72h after = highest multiplier
    post_cranial_surgery_hours: float = 0.0
    peripheral_neuropathy: bool = False
    vestibular_disorder: bool = False   # BPPV, Menière's
    epilepsy: bool = False
    dementia: bool = False
    muscular_dystrophy: bool = False
    inflammatory_myopathy: bool = False  # polymyositis, dermatomyositis

    # ── Cardiovascular
    cardiac_arrhythmia: bool = False     # AF, QTc prolonged, brady
    orthostatic_hypotension: bool = False
    heart_failure: bool = False
    ecg_qtc_prolonged: bool = False

    # ── Surgical / procedural
    post_general_surgery: bool = False   # <30 days
    post_ortho_surgery: bool = False     # <30 days
    post_cardiac_surgery: bool = False   # <72h highest, then tapers
    post_surgery_hours: float = 0.0

    # ── Obstetric
    pregnancy_trimester: int = 0         # 0=not pregnant, 1/2/3
    post_partum_days: int = 0            # 0=not applicable

    # ── Functional
    vision_impaired: bool = False        # acuity <6/12 or field defect
    sarcopenia: bool = False             # EWGSOP2 criteria
    tug_time_s: float = 0.0             # physician-documented TUG
    prior_fall_history: bool = False
    fall_count_last_3m: int = 0
    uses_ambulatory_aid: str = "none"   # none / cane_crutch / walker / furniture
    iv_line: bool = False
    mental_status: str = "oriented"     # oriented / overestimates_ability / confused

    # ── Pharmacological (FRID classification from BMC Geriatrics 2023)
    # Category 1 — CNS (psychotropic FRIDs) — strong evidence
    on_opioids: bool = False
    on_antipsychotics: bool = False
    on_antidepressants: bool = False
    on_benzodiazepines: bool = False
    on_antiparkinson: bool = False       # AOR 1.30 — highest FRID
    on_antiepileptics: bool = False
    # Category 2 — CVD FRIDs — moderate evidence
    on_antihypertensives: bool = False
    on_diuretics: bool = False
    on_beta_blockers: bool = False
    on_alpha_blockers: bool = False      # tamsulosin etc.
    # Other FRIDs
    on_nsaids: bool = False
    on_muscle_relaxants: bool = False

    # ── Diagnosis context (for IPSG.6 fall category assignment)
    ward_type: str = "general"           # general / icu / neuro / obstetric / physio

    @property
    def bmi(self) -> float:
        h = self.height_cm / 100.0
        return self.weight_kg / (h * h) if h > 0 else 0.0

    @property
    def frid_cat1_count(self) -> int:
        return sum([self.on_opioids, self.on_antipsychotics, self.on_antidepressants,
                    self.on_benzodiazepines, self.on_antiparkinson, self.on_antiepileptics])

    @property
    def frid_cat2_count(self) -> int:
        return sum([self.on_antihypertensives, self.on_diuretics,
                    self.on_beta_blockers, self.on_alpha_blockers])

    @property
    def total_medications(self) -> int:
        return self.frid_cat1_count + self.frid_cat2_count + \
               int(self.on_nsaids) + int(self.on_muscle_relaxants)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: IMU READING — one timestamp window of sensor data
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IMUReading:
    timestamp: float          # epoch seconds
    acc_x: float              # m/s² — lateral (mediolateral)
    acc_y: float              # m/s² — sagittal (anteroposterior)
    acc_z: float              # m/s² — vertical
    gyro_x: float             # rad/s — pitch rate
    gyro_y: float             # rad/s — roll rate
    gyro_z: float             # rad/s — yaw rate
    spo2: Optional[float] = None    # %
    heart_rate: Optional[float] = None  # bpm
    systolic_bp: Optional[float] = None # mmHg


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: COMPUTED FEATURES — per-window feature extraction
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WindowFeatures:
    timestamp: float
    pitch_deg: float          # sagittal trunk tilt (forward/back) — atan2
    roll_deg: float           # lateral trunk tilt — atan2
    tilt_magnitude: float     # combined tilt = sqrt(pitch² + roll²)
    smv: float                # signal magnitude vector (m/s²) — √(ax²+ay²+az²)
    jerk: float               # rate of change of SMV (m/s³)
    tilt_rate: float          # °/s — delta tilt_magnitude / delta time
    gyro_magnitude: float     # √(gx²+gy²+gz²) rad/s
    cop_x: float              # estimated Centre of Pressure lateral (mm)
    cop_y: float              # estimated Centre of Pressure sagittal (mm)
    spo2: Optional[float] = None
    heart_rate: Optional[float] = None
    systolic_bp: Optional[float] = None


def extract_features(
    reading: IMUReading,
    prev_features: Optional[WindowFeatures],
    sensor_height_m: float = 1.0   # lumbar sensor ≈ 1.0 m from ground
) -> WindowFeatures:
    """
    Extract biomechanical features from one IMU timestamp.
    Formulas:
      pitch = atan2(acc_x, √(acc_y²+acc_z²))  — sagittal tilt
      roll  = atan2(acc_y, √(acc_x²+acc_z²))  — lateral tilt
      CoP_x ≈ (h × acc_x) / g                 — SDI estimation
      CoP_y ≈ (h × acc_y) / g
    """
    g = 9.81
    dt = (reading.timestamp - prev_features.timestamp) if prev_features else 0.02

    pitch = math.degrees(math.atan2(
        reading.acc_x,
        math.sqrt(reading.acc_y**2 + reading.acc_z**2)
    ))
    roll = math.degrees(math.atan2(
        reading.acc_y,
        math.sqrt(reading.acc_x**2 + reading.acc_z**2)
    ))

    # Low-pass filter (EMA) to reject high-frequency footstep acceleration 
    # spikes during gait, preventing false "high tilt" readings while walking.
    if prev_features:
        alpha = 0.15
        pitch = (alpha * pitch) + ((1 - alpha) * prev_features.pitch_deg)
        roll = (alpha * roll) + ((1 - alpha) * prev_features.roll_deg)

    tilt_mag = math.sqrt(pitch**2 + roll**2)
    smv = math.sqrt(reading.acc_x**2 + reading.acc_y**2 + reading.acc_z**2)
    gyro_mag = math.sqrt(reading.gyro_x**2 + reading.gyro_y**2 + reading.gyro_z**2)

    jerk = abs(smv - prev_features.smv) / dt if prev_features and dt > 0 else 0.0
    tilt_rate = abs(tilt_mag - prev_features.tilt_magnitude) / dt if prev_features and dt > 0 else 0.0

    # CoP estimate (mm) — SDI computation uses these time series
    cop_x = (sensor_height_m * reading.acc_x / g) * 1000
    cop_y = (sensor_height_m * reading.acc_y / g) * 1000

    return WindowFeatures(
        timestamp=reading.timestamp,
        pitch_deg=pitch,
        roll_deg=roll,
        tilt_magnitude=tilt_mag,
        smv=smv,
        jerk=jerk,
        tilt_rate=tilt_rate,
        gyro_magnitude=gyro_mag,
        cop_x=cop_x,
        cop_y=cop_y,
        spo2=reading.spo2,
        heart_rate=reading.heart_rate,
        systolic_bp=reading.systolic_bp,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: FALL vs INTENTIONAL BEND DISCRIMINATOR
# This is the core clinical differentiator. Reads a short rolling buffer.
# ─────────────────────────────────────────────────────────────────────────────

class MotionDiscriminator:
    """
    Temporal profile-based discriminator.

    Key clinical logic:
      - FALL:   rapid tilt onset (<0.8 s to peak), high jerk, NO recovery in 8 s
      - BEND:   slow tilt onset (>2.0 s to peak), low jerk, returns upward
      - NEURO:  slow rate (18–30°/s) but NO recovery — Parkinson's slow fall

    Uses a 10-second rolling buffer of WindowFeatures.
    """

    def __init__(self, buffer_seconds: float = 10.0):
        self._buf: deque = deque()
        self._buf_seconds = buffer_seconds
        self._event_start_time: Optional[float] = None
        self._event_start_tilt: float = 0.0
        self._peak_tilt: float = 0.0
        self._peak_tilt_time: Optional[float] = None
        self._impact_detected: bool = False

    def push(self, f: WindowFeatures) -> None:
        self._buf.append(f)
        # Prune old entries
        cutoff = f.timestamp - self._buf_seconds
        while self._buf and self._buf[0].timestamp < cutoff:
            self._buf.popleft()

        # Track peak tilt
        if f.tilt_magnitude > self._peak_tilt:
            self._peak_tilt = f.tilt_magnitude
            self._peak_tilt_time = f.timestamp

        # Detect impact
        if f.smv >= Thresholds.POST_IMPACT_SMV and f.jerk >= Thresholds.JERK_WARN:
            self._impact_detected = True

    def _tilt_has_recovered(self, current: WindowFeatures) -> bool:
        """True if tilt has dropped by ≥15° from peak toward baseline."""
        if self._peak_tilt == 0:
            return False
        return (self._peak_tilt - current.tilt_magnitude) >= Thresholds.RECOVERY_THRESHOLD_DEG

    def _time_since_peak(self, current: WindowFeatures) -> float:
        if self._peak_tilt_time is None:
            return 0.0
        return current.timestamp - self._peak_tilt_time

    def classify(
        self,
        current: WindowFeatures,
        profile: PatientProfile,
        sdi: float
    ) -> MotionEvent:
        """
        Classify the current motion event using the 5-feature temporal profile.
        """
        T = Thresholds
        tilt = current.tilt_magnitude
        rate = current.tilt_rate
        jerk = current.jerk
        recovered = self._tilt_has_recovered(current)
        time_since_peak = self._time_since_peak(current)

        # ── 1. STABLE — nothing happening ────────────────────────────────────
        # If the sensor is practically motionless, the patient is either lying down, 
        # sitting still, or the sensor is on a table. Absolute angle (tilt) does not matter.
        if rate < 5.0 and jerk < 1.5:
            self._reset_tracking()
            return MotionEvent.STABLE

        # If they are moving slowly within safe upright boundaries
        if tilt < T.TILT_WARN_DEG and rate < T.BEND_MAX_TILT_RATE and jerk < T.BEND_MAX_JERK:
            self._reset_tracking()
            return MotionEvent.STABLE

        # ── 2. CONFIRMED FALL — impact + no recovery ──────────────────────────
        if self._impact_detected and time_since_peak > T.RECOVERY_WINDOW_S and not recovered:
            return MotionEvent.CONFIRMED_FALL

        # ── 3. FALL TRAJECTORY — rapid uncontrolled tilt ─────────────────────
        if rate >= T.FALL_MIN_TILT_RATE and jerk >= T.JERK_WARN and tilt >= T.TILT_WARN_DEG:
            # Check time-to-peak — if reached peak tilt quickly, it's a fall
            if self._peak_tilt_time and (self._peak_tilt_time - (current.timestamp - 2.0)) < T.FALL_MAX_TIME_TO_PEAK:
                return MotionEvent.FALL_TRAJECTORY

        # ── 4. SLOW NEUROLOGICAL FALL (Parkinson's / cerebellar) ─────────────
        # Slow rate but SDI < threshold and no recovery = neuro fall
        neuro_patient = profile.parkinsons or profile.post_stroke or profile.vestibular_disorder
        if (neuro_patient or sdi < T.SDI_NEURO_THRESHOLD) and \
           rate >= T.SLOW_NEURO_TILT_RATE and \
           tilt >= T.TILT_WARN_DEG and \
           time_since_peak > T.RECOVERY_WINDOW_S and \
           not recovered:
            return MotionEvent.SLOW_NEURO_FALL

        # ── 5. INTENTIONAL BEND — slow controlled descent ─────────────────────
        # Rate is low, jerk is low, tilt can be high but arrived slowly
        if rate < T.BEND_MAX_TILT_RATE and jerk < T.BEND_MAX_JERK:
            # Check if patient eventually recovers (upward return expected)
            if time_since_peak < T.RECOVERY_WINDOW_S:
                return MotionEvent.INTENTIONAL_BEND  # still in the bend, monitoring

        # ── 6. NO RECOVERY AFTER SLOW BEND — escalate ────────────────────────
        if rate < T.FALL_MIN_TILT_RATE and jerk < T.JERK_CRITICAL and \
           tilt >= T.TILT_WARN_DEG and \
           time_since_peak > T.RECOVERY_WINDOW_S and \
           not recovered:
            # Slow onset but never came back = likely fall for neuro patient
            if neuro_patient:
                return MotionEvent.SLOW_NEURO_FALL
            return MotionEvent.PRE_FALL_WARNING  # escalate to caution

        # ── 7. POSTURAL LEAN — sustained elevated tilt, slow rate ─────────────
        # Exclude high-jerk events (like walking) from triggering static postural lean
        if tilt >= T.TILT_WARN_DEG and rate < T.FALL_MIN_TILT_RATE and jerk < T.JERK_WARN:
            return MotionEvent.POSTURAL_LEAN

        # ── 8. PRE-FALL WARNING — intermediate pattern ───────────────
        # Requires actual sudden angular motion, preventing false alarms from simply sitting
        if rate >= T.SLOW_NEURO_TILT_RATE and jerk >= T.JERK_WARN:
            return MotionEvent.PRE_FALL_WARNING

        return MotionEvent.STABLE

    def _reset_tracking(self):
        self._peak_tilt = 0.0
        self._peak_tilt_time = None
        self._impact_detected = False


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: SDI CALCULATOR — Sway Density Index from rolling CoP trace
# ─────────────────────────────────────────────────────────────────────────────

class SDICalculator:
    """
    Computes Sway Density Index (peaks/s) from CoP_x time series.
    SDI = number of local maxima in the CoP_x signal per second.
    Normal: >2.0; Elderly declining: 1.0–2.0; Neurological deficit: <1.0
    Source: Mancini & Horak 2010; SDI/MPI literature.
    """

    def __init__(self, window_s: float = 2.0):
        self._cop_buf: deque = deque()
        self._window_s = window_s

    def push(self, f: WindowFeatures) -> None:
        self._cop_buf.append((f.timestamp, f.cop_x))
        cutoff = f.timestamp - self._window_s
        while self._cop_buf and self._cop_buf[0][0] < cutoff:
            self._cop_buf.popleft()

    def compute_sdi(self) -> float:
        if len(self._cop_buf) < 5:
            return 2.5  # default healthy
        vals = [v for _, v in self._cop_buf]
        # Count zero-crossing reversals (local extrema)
        peaks = 0
        for i in range(1, len(vals) - 1):
            if (vals[i] > vals[i-1] and vals[i] > vals[i+1]) or \
               (vals[i] < vals[i-1] and vals[i] < vals[i+1]):
                peaks += 1
        elapsed = self._cop_buf[-1][0] - self._cop_buf[0][0]
        return peaks / elapsed if elapsed > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: MORSE FALL SCALE — static score from patient profile
# ─────────────────────────────────────────────────────────────────────────────

def compute_mfs_score(p: PatientProfile) -> int:
    """
    Morse Fall Scale — Morse et al. 1989.
    Max score 125; high risk ≥45 (or ≥40 for obstetric patients).
    """
    score = 0

    # 1. History of falls (past 3 months) — 25 pts, strongest single predictor
    if p.prior_fall_history or p.fall_count_last_3m > 0:
        score += 25

    # 2. Secondary diagnoses (≥2 active medical diagnoses) — 15 pts
    diagnoses = sum([
        p.parkinsons, p.post_stroke, p.cardiac_arrhythmia, p.heart_failure,
        p.peripheral_neuropathy, p.vestibular_disorder, p.epilepsy,
        p.dementia, p.muscular_dystrophy, p.orthostatic_hypotension
    ])
    if diagnoses >= 2:
        score += 15

    # 3. Ambulatory aid — 0 / 15 / 30 pts
    aid_scores = {"none": 0, "nurse": 0, "bed_rest": 0,
                  "cane": 15, "crutch": 15, "walker": 15,
                  "furniture": 30, "wall": 30}
    score += aid_scores.get(p.uses_ambulatory_aid, 0)

    # 4. IV line / heparin lock — 20 pts
    if p.iv_line:
        score += 20

    # 5. Gait assessment — 0 / 10 / 20 pts
    if p.tug_time_s >= Thresholds.TUG_HIGH_RISK_S:
        score += 20
    elif p.tug_time_s > 0 and p.tug_time_s >= 8.0:
        score += 10

    # 6. Mental status — 0 / 15 pts
    if p.mental_status in ("overestimates_ability", "confused", "delirious"):
        score += 15

    return score


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: FRID SCORE — pharmacological fall risk accumulator
# ─────────────────────────────────────────────────────────────────────────────

def compute_frid_score(p: PatientProfile) -> float:
    """
    Fall-Risk Increasing Drug score.
    Cat 1 (CNS/psychotropic): AOR 1.14–1.30 per drug (BMC Geriatrics 2023).
    Cat 2 (CVD): moderate evidence, JAMDA 2024 longitudinal.
    Polypharmacy ≥3 = independent multiplier.
    Returns additive risk points (not a multiplier at this stage).
    """
    score = 0.0

    # Category 1 — CNS FRIDs (strongest evidence)
    if p.on_antiparkinson:   score += 18.0  # AOR 1.30 — highest
    if p.on_opioids:         score += 14.0  # AOR 1.23
    if p.on_antiepileptics:  score += 12.0  # AOR 1.16
    if p.on_antipsychotics:  score += 10.0  # AOR 1.14, dose-response HR 2.78
    if p.on_antidepressants: score += 10.0  # AOR 1.10 — strongest in Leipzig 1999
    if p.on_benzodiazepines: score += 10.0  # AOR 1.06 + CNS depression

    # Category 2 — CVD FRIDs (moderate evidence)
    if p.on_antihypertensives: score += 7.0
    if p.on_diuretics:         score += 6.0  # electrolyte/volume depletion
    if p.on_beta_blockers:     score += 5.0
    if p.on_alpha_blockers:    score += 6.0  # tamsulosin — significant (Welk 2015)

    # Other FRIDs
    if p.on_nsaids:           score += 4.0
    if p.on_muscle_relaxants: score += 6.0  # direct neuromuscular blockade

    # Polypharmacy compounding — Leipzig 1999: risk increases linearly with drugs
    if p.total_medications >= 3 and (p.frid_cat1_count > 0 or p.frid_cat2_count > 0):
        score *= 1.4  # polypharmacy multiplier when any FRID present

    return score


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10: AGE AND CONDITION MULTIPLIERS
# Source: MFS age-stratified AUC data; WHO falls epidemiology; condition literature
# ─────────────────────────────────────────────────────────────────────────────

def compute_age_multiplier(age: int) -> float:
    """Age cohort multiplier — MFS AUC 0.837 in ≥75 group."""
    if age < 45:  return 1.0
    if age < 65:  return 1.3
    if age < 75:  return 1.6
    return 2.0


def compute_bmi_multiplier(bmi: float) -> float:
    """
    BMI-based risk multiplier.
    Sarcopenia prevalence: normal=24.8%, overweight=37.0%, obese=54.3% (AHA 2024).
    """
    if bmi < 18.5:  return 1.3   # underweight — malnutrition/sarcopenia
    if bmi < 25.0:  return 1.0   # reference
    if bmi < 30.0:  return 1.15  # overweight — rising sarcopenia
    if bmi < 40.0:  return 1.4   # obese — sarcopenic obesity risk
    return 1.7                    # morbid obesity — extreme CoM displacement


def compute_condition_multiplier(p: PatientProfile) -> float:
    """
    Largest single condition multiplier dominates.
    Source: condition-specific literature as documented in research database.
    """
    mult = 1.0

    # Neurological — highest multipliers
    if p.post_cranial_surgery:
        # Acute phase ≤72h = 2.5×, then tapers
        if p.post_cranial_surgery_hours <= 72:
            mult = max(mult, 2.5)
        elif p.post_cranial_surgery_hours <= 168:  # 7 days
            mult = max(mult, 2.0)
        else:
            mult = max(mult, 1.6)

    if p.parkinsons:              mult = max(mult, 2.2)
    if p.muscular_dystrophy:      mult = max(mult, 2.1)
    if p.post_stroke:             mult = max(mult, 2.0)
    if p.inflammatory_myopathy:   mult = max(mult, 1.9)
    if p.vestibular_disorder:     mult = max(mult, 1.8)
    if p.epilepsy:                mult = max(mult, 1.7)  # unanticipated physiological
    if p.cardiac_arrhythmia or p.ecg_qtc_prolonged: mult = max(mult, 1.8)
    if p.orthostatic_hypotension: mult = max(mult, 1.7)
    if p.peripheral_neuropathy:   mult = max(mult, 1.6)
    if p.dementia:                mult = max(mult, 1.6)
    if p.vision_impaired:         mult = max(mult, 1.4)
    if p.sarcopenia:              mult = max(mult, 1.8)  # sarcopenic obesity if also BMI≥30

    # Surgical — post-operative
    if p.post_cardiac_surgery:
        if p.post_surgery_hours <= 72:
            mult = max(mult, 2.5)
        else:
            mult = max(mult, 1.8)
    if p.post_ortho_surgery:   mult = max(mult, 2.0)
    if p.post_general_surgery: mult = max(mult, 1.8)

    # Obstetric
    trimester_mult = {1: 1.2, 2: 1.5, 3: 1.8}
    if p.pregnancy_trimester in trimester_mult:
        mult = max(mult, trimester_mult[p.pregnancy_trimester])
    if p.post_partum_days > 0 and p.post_partum_days <= 42:
        mult = max(mult, 1.4)

    # Sarcopenic obesity compounding (AHA Circulation 2024)
    if p.sarcopenia and p.bmi >= 30:
        mult = max(mult, 2.0)

    return mult


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11: DYNAMIC SENSOR SCORING — per-window biomechanical score
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DynamicScoreResult:
    raw_score: float
    alerts: list
    dominant_trigger: str
    motion_event: MotionEvent
    sdi: float


def compute_dynamic_score(
    f: WindowFeatures,
    motion_event: MotionEvent,
    sdi: float,
    profile: PatientProfile
) -> DynamicScoreResult:
    """
    Score the current sensor window against all biomechanical thresholds.
    Returns a raw dynamic score (0–100) before static profile multiplication.
    """
    T = Thresholds
    score = 0.0
    alerts = []
    dominant = "stable"

    # ── MOTION EVENT SCORES ───────────────────────────────────────────────────
    if motion_event == MotionEvent.CONFIRMED_FALL:
        score = 100.0
        alerts.append("CONFIRMED FALL — impact detected, no recovery")
        dominant = "confirmed_fall"

    elif motion_event in (MotionEvent.FALL_TRAJECTORY, MotionEvent.SLOW_NEURO_FALL):
        score = 85.0
        ev = "slow neurological" if motion_event == MotionEvent.SLOW_NEURO_FALL else "rapid"
        alerts.append(f"ACTIVE FALL TRAJECTORY ({ev}) — no recovery detected")
        dominant = "fall_trajectory"

    elif motion_event == MotionEvent.INTENTIONAL_BEND:
        # Intentional bend: suppress fall alert, but flag for high-risk patients
        score = 5.0
        if profile.parkinsons or profile.vestibular_disorder:
            score = 20.0
            alerts.append("Intentional bend noted — elevated monitoring for neuro patient")
        dominant = "intentional_bend"

    elif motion_event == MotionEvent.POSTURAL_LEAN:
        score = 30.0
        alerts.append(f"Sustained postural lean — tilt {f.tilt_magnitude:.1f}°")
        dominant = "postural_lean"

    elif motion_event == MotionEvent.PRE_FALL_WARNING:
        score = 45.0
        alerts.append("Pre-fall warning — prodromal instability pattern")
        dominant = "prodromal"

    else:
        # STABLE — still check individual thresholds
        # Tilt score
        if f.tilt_magnitude >= T.TILT_CRITICAL_DEG:
            score += 30.0; alerts.append(f"Critical trunk tilt {f.tilt_magnitude:.1f}°")
            dominant = "tilt_critical"
        elif f.tilt_magnitude >= T.TILT_WARN_DEG:
            score += 18.0; alerts.append(f"Trunk tilt {f.tilt_magnitude:.1f}° (warn)")
            if dominant == "stable": dominant = "tilt_warn"

        # SMV score
        if f.smv >= T.SMV_CRITICAL:
            score += 25.0; alerts.append(f"Critical SMV {f.smv:.1f} m/s²")
            if dominant == "stable": dominant = "smv_critical"
        elif f.smv >= T.SMV_WARN:
            score += 15.0; alerts.append(f"Elevated SMV {f.smv:.1f} m/s²")
            if dominant == "stable": dominant = "smv_warn"

        # Jerk
        if f.jerk >= T.JERK_CRITICAL:
            score += 20.0; alerts.append(f"Critical jerk {f.jerk:.1f} m/s³ — impact-level")
            if dominant == "stable": dominant = "jerk"
        elif f.jerk >= T.JERK_WARN:
            score += 10.0; alerts.append(f"Elevated jerk {f.jerk:.1f} m/s³")

        # Pitch / roll
        if abs(f.pitch_deg) >= T.PITCH_WARN_DEG:
            score += 8.0; alerts.append(f"Sagittal pitch {f.pitch_deg:.1f}°")
        if abs(f.roll_deg) >= T.ROLL_WARN_DEG:
            score += 8.0; alerts.append(f"Lateral roll {f.roll_deg:.1f}°")

    # ── PHYSIOLOGICAL SCORES ──────────────────────────────────────────────────
    if f.spo2 is not None:
        if f.spo2 <= T.SPO2_CRITICAL:
            score += 22.0
            alerts.append(f"SpO2 CRITICAL {f.spo2:.0f}% — hypoxaemia, syncope risk")
            if dominant == "stable": dominant = "spo2_critical"
        elif f.spo2 <= T.SPO2_WARN:
            score += 12.0
            alerts.append(f"SpO2 {f.spo2:.0f}% — cerebral hypoperfusion warning")

    # ── SDI NEUROLOGICAL SCORE ────────────────────────────────────────────────
    if sdi < T.SDI_CRITICAL:
        score += 18.0
        alerts.append(f"SDI critically low {sdi:.2f} peaks/s — neurological postural failure")
        if dominant == "stable": dominant = "sdi_critical"
    elif sdi < T.SDI_WARN:
        score += 8.0
        alerts.append(f"SDI reduced {sdi:.2f} peaks/s — declining postural control")

    return DynamicScoreResult(
        raw_score=min(score, 100.0),
        alerts=alerts,
        dominant_trigger=dominant,
        motion_event=motion_event,
        sdi=sdi
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12: COMPOSITE RISK SCORE — full algorithm integration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CompositeRiskResult:
    # Scores
    mfs_score: int
    frid_score: float
    dynamic_score: float
    age_multiplier: float
    bmi_multiplier: float
    condition_multiplier: float
    composite_score: float           # 0–100 final

    # State
    postural_state: PosturalState
    motion_event: MotionEvent
    fall_category: FallCategory
    sdi: float

    # Output
    alerts: list
    dominant_trigger: str
    risk_level: str                  # LOW / MEDIUM / HIGH / CRITICAL

    # Context
    timestamp: float
    patient_id: str


def compute_composite_risk(
    profile: PatientProfile,
    dynamic: DynamicScoreResult,
    mfs_score: int,
    frid_score: float,
    consecutive_warn_windows: int
) -> CompositeRiskResult:
    """
    Final composite score formula:
      composite = (static_base + dynamic_raw) × age_mult × bmi_mult × condition_mult
      capped at 100.

    static_base = MFS contribution (0–40 pts mapped from 0–125 MFS) + FRID score (0–40)
    dynamic_raw = per-window biomechanical score (0–100)

    The condition multiplier is applied LAST and represents the irreducible
    neurological/physiological amplification of any event.
    """
    T = Thresholds

    # Map MFS 0–125 to 0–40 contribution
    mfs_contribution = min((mfs_score / 125.0) * 40.0, 40.0)

    # FRID capped at 40 contribution
    frid_contribution = min(frid_score, 40.0)

    # Static base (before dynamic events)
    static_base = mfs_contribution + frid_contribution

    age_mult = compute_age_multiplier(profile.age)
    bmi_mult = compute_bmi_multiplier(profile.bmi)
    condition_mult = compute_condition_multiplier(profile)

    # Dynamic events bypass static scoring for emergencies
    if dynamic.motion_event == MotionEvent.CONFIRMED_FALL:
        composite = 100.0
    elif dynamic.motion_event in (MotionEvent.FALL_TRAJECTORY, MotionEvent.SLOW_NEURO_FALL):
        # Active fall: dynamic dominates but condition mult still applies
        composite = min(dynamic.raw_score * condition_mult, 100.0)
    elif dynamic.motion_event == MotionEvent.INTENTIONAL_BEND:
        # Suppress false alarm — use only static score
        composite = min(static_base * age_mult * 0.3, 40.0)
    else:
        # Normal composite: static + dynamic, all multipliers
        combined = (static_base * 0.4) + (dynamic.raw_score * 0.6)
        composite = min(combined * age_mult * bmi_mult * condition_mult, 100.0)

    # ── POSTURAL STATE (temporal state machine) ───────────────────────────────
    if dynamic.motion_event == MotionEvent.CONFIRMED_FALL:
        postural_state = PosturalState.ACTIVE_FALL
    elif composite >= 60.0:
        postural_state = PosturalState.ACTIVE_FALL
    elif composite >= 30.0 and consecutive_warn_windows >= 3:
        postural_state = PosturalState.PRODROMAL
    else:
        postural_state = PosturalState.HOMEOSTASIS

    # ── FALL CATEGORY (IPSG.6 Quigley framework) ──────────────────────────────
    if dynamic.motion_event == MotionEvent.INTENTIONAL_BEND:
        fall_cat = FallCategory.NOT_A_FALL
    elif dynamic.motion_event in (MotionEvent.CONFIRMED_FALL, MotionEvent.FALL_TRAJECTORY):
        # Classify as anticipated vs unanticipated
        unanticipated_triggers = [
            profile.cardiac_arrhythmia, profile.epilepsy,
            profile.orthostatic_hypotension, profile.ecg_qtc_prolonged
        ]
        if any(unanticipated_triggers):
            fall_cat = FallCategory.UNANTICIPATED_PHYSIO
        elif mfs_score >= T.MFS_HIGH_RISK:
            fall_cat = FallCategory.ANTICIPATED_PHYSIO
        else:
            fall_cat = FallCategory.ACCIDENTAL
    else:
        fall_cat = FallCategory.ANTICIPATED_PHYSIO if mfs_score >= T.MFS_MEDIUM_RISK \
                   else FallCategory.ACCIDENTAL

    # ── RISK LEVEL ────────────────────────────────────────────────────────────
    if composite >= 75:   risk_level = "CRITICAL"
    elif composite >= 60: risk_level = "HIGH"
    elif composite >= 35: risk_level = "MEDIUM"
    else:                 risk_level = "LOW"

    return CompositeRiskResult(
        mfs_score=mfs_score,
        frid_score=frid_score,
        dynamic_score=dynamic.raw_score,
        age_multiplier=age_mult,
        bmi_multiplier=bmi_mult,
        condition_multiplier=condition_mult,
        composite_score=round(composite, 1),
        postural_state=postural_state,
        motion_event=dynamic.motion_event,
        fall_category=fall_cat,
        sdi=dynamic.sdi,
        alerts=dynamic.alerts,
        dominant_trigger=dynamic.dominant_trigger,
        risk_level=risk_level,
        timestamp=time.time(),
        patient_id=profile.patient_id
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13: PATIENT ENGINE — per-patient stateful processor
# ─────────────────────────────────────────────────────────────────────────────

class PatientEngine:
    """
    Stateful per-patient fall risk processor.
    Maintains rolling window, discriminator, SDI calculator,
    consecutive-warning counter, and baseline calibration.
    """

    CALIBRATION_SECONDS = 30.0   # first 30 s = baseline calibration per session

    def __init__(self, profile: PatientProfile):
        self.profile = profile
        self.mfs_score = compute_mfs_score(profile)
        self.frid_score = compute_frid_score(profile)

        self._discriminator = MotionDiscriminator()
        self._sdi_calc = SDICalculator()
        self._prev_features: Optional[WindowFeatures] = None
        self._consecutive_warn = 0
        self._session_start: Optional[float] = None
        self._baseline_smv: Optional[float] = None
        self._baseline_tilt: Optional[float] = None
        self._calibration_readings: list = []
        self._last_result: Optional[CompositeRiskResult] = None

    def _is_calibrating(self, timestamp: float) -> bool:
        if self._session_start is None:
            self._session_start = timestamp
            return True
        return (timestamp - self._session_start) < self.CALIBRATION_SECONDS

    def _finalise_calibration(self):
        if not self._calibration_readings:
            return
        smvs = [f.smv for f in self._calibration_readings]
        tilts = [f.tilt_magnitude for f in self._calibration_readings]
        self._baseline_smv = sum(smvs) / len(smvs)
        self._baseline_tilt = sum(tilts) / len(tilts)
        self._calibration_readings.clear()

    def process(self, reading: IMUReading) -> CompositeRiskResult:
        """Process one IMU reading and return the current composite risk."""

        # 1. Extract features
        features = extract_features(reading, self._prev_features)
        self._prev_features = features

        # 2. Calibration phase — collect baseline, return low-risk result
        if self._is_calibrating(reading.timestamp):
            self._calibration_readings.append(features)
            return CompositeRiskResult(
                mfs_score=self.mfs_score, frid_score=self.frid_score,
                dynamic_score=0.0, age_multiplier=1.0,
                bmi_multiplier=1.0, condition_multiplier=1.0,
                composite_score=0.0,
                postural_state=PosturalState.HOMEOSTASIS,
                motion_event=MotionEvent.STABLE,
                fall_category=FallCategory.ACCIDENTAL,
                sdi=2.5, alerts=["Calibrating patient baseline..."],
                dominant_trigger="calibration", risk_level="LOW",
                timestamp=reading.timestamp, patient_id=self.profile.patient_id
            )

        # Finalise calibration on first post-calibration reading
        if self._baseline_smv is None:
            self._finalise_calibration()

        # 3. Update SDI and discriminator
        self._sdi_calc.push(features)
        sdi = self._sdi_calc.compute_sdi()
        self._discriminator.push(features)
        motion_event = self._discriminator.classify(features, self.profile, sdi)

        # 4. Dynamic score
        dynamic = compute_dynamic_score(features, motion_event, sdi, self.profile)

        # 5. Consecutive warning window counter (temporal state machine)
        if dynamic.raw_score >= 30:
            self._consecutive_warn += 1
        else:
            self._consecutive_warn = 0

        # 6. Composite score
        result = compute_composite_risk(
            self.profile, dynamic,
            self.mfs_score, self.frid_score,
            self._consecutive_warn
        )
        self._last_result = result
        return result


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14: GAIT METRICS — step count, speed, TUG from raw accelerometer
# ─────────────────────────────────────────────────────────────────────────────

class GaitAnalyser:
    """
    Derives clinical gait metrics from raw IMU accelerometer data.

    Step detection:  vertical acc_z peaks above threshold, min 0.3 s apart.
    Gait speed:      displacement per step estimated from stride length proxy.
    TUG detection:   3-phase pattern — sit-to-stand → walk → return-to-sit.
    """

    STEP_THRESHOLD = 1.5   # m/s² above mean — footstrike impact
    MIN_STEP_INTERVAL = 0.3  # seconds — minimum inter-step interval
    TUG_STAND_SMV = 13.0   # m/s² — sit-to-stand SMV signature
    TUG_SIT_SMV = 12.0     # m/s² — return-to-sit SMV signature

    def __init__(self):
        self._acc_z_buf: deque = deque(maxlen=500)
        self._step_times: deque = deque(maxlen=200)
        self._total_steps: int = 0
        self._smv_buf: deque = deque(maxlen=200)
        self._tug_start: Optional[float] = None
        self._tug_phase: str = "idle"  # idle → standing → walking → returning → done
        self._tug_times: deque = deque(maxlen=20)

    def push(self, reading: IMUReading) -> None:
        self._acc_z_buf.append((reading.timestamp, reading.acc_z))
        smv = math.sqrt(reading.acc_x**2 + reading.acc_y**2 + reading.acc_z**2)
        self._smv_buf.append((reading.timestamp, smv))
        self._detect_step(reading.timestamp, reading.acc_z)
        self._detect_tug_phase(reading.timestamp, smv)

    def _detect_step(self, t: float, acc_z: float) -> None:
        """Peak detection in vertical acceleration for step counting."""
        if len(self._acc_z_buf) < 5:
            return
        recent = [v for _, v in list(self._acc_z_buf)[-10:]]
        mean_z = sum(recent) / len(recent)
        # Detect local peak above threshold
        if acc_z > mean_z + self.STEP_THRESHOLD:
            if not self._step_times or (t - self._step_times[-1]) > self.MIN_STEP_INTERVAL:
                self._step_times.append(t)
                self._total_steps += 1

    def _detect_tug_phase(self, t: float, smv: float) -> None:
        """TUG time auto-detection using SMV phase signatures."""
        if self._tug_phase == "idle":
            if smv >= self.TUG_STAND_SMV:
                self._tug_phase = "standing"
                self._tug_start = t
        elif self._tug_phase == "standing":
            if smv < self.TUG_STAND_SMV * 0.7:
                self._tug_phase = "walking"
        elif self._tug_phase == "walking":
            if smv >= self.TUG_SIT_SMV and self._tug_start is not None:
                self._tug_phase = "returning"
        elif self._tug_phase == "returning":
            if smv < self.TUG_SIT_SMV * 0.6 and self._tug_start is not None:
                tug_time = t - self._tug_start
                if 3.0 < tug_time < 120.0:  # sanity range
                    self._tug_times.append(tug_time)
                self._tug_phase = "idle"
                self._tug_start = None

    @property
    def step_count(self) -> int:
        return self._total_steps

    @property
    def gait_speed_ms(self) -> Optional[float]:
        """Gait speed in m/s — requires ≥4 steps for meaningful estimate."""
        if len(self._step_times) < 4:
            return None
        # Stride duration = mean inter-step interval
        intervals = [self._step_times[i+1] - self._step_times[i]
                     for i in range(len(self._step_times)-1)]
        mean_stride_s = sum(intervals) / len(intervals)
        # Typical stride length 0.6–1.4 m; estimate from cadence
        # Approximate: speed ≈ stride_length / stride_time, stride ≈ 2 × step_length
        # Step length proxy: 0.28 × height (empirical, Grieve & Gear 1966)
        return round(1.0 / mean_stride_s * 0.35, 2)  # 0.35 m average step length proxy

    @property
    def last_tug_time(self) -> Optional[float]:
        return round(self._tug_times[-1], 1) if self._tug_times else None

    @property
    def tug_risk_level(self) -> str:
        t = self.last_tug_time
        if t is None: return "not_measured"
        if t >= Thresholds.TUG_HIGH_RISK_S: return "high_risk"
        if t >= 8.0: return "moderate_risk"
        return "low_risk"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 15: WARD MANAGER — multi-patient coordinator
# ─────────────────────────────────────────────────────────────────────────────

class WardManager:
    """
    Manages all patient engines on a ward.
    Single entry point for the FastAPI backend to call.
    """

    def __init__(self):
        self._engines: dict[str, PatientEngine] = {}
        self._gait: dict[str, GaitAnalyser] = {}

    def admit_patient(self, profile: PatientProfile) -> dict:
        """Admit a new patient and compute their static baseline scores."""
        engine = PatientEngine(profile)
        self._engines[profile.patient_id] = engine
        self._gait[profile.patient_id] = GaitAnalyser()

        mfs = engine.mfs_score
        frid = engine.frid_score

        # IPSG.6 ward: use adjusted cutoff for obstetric patients
        cutoff = Thresholds.MFS_OB_GYN_HIGH if profile.ward_type == "obstetric" \
                 else Thresholds.MFS_HIGH_RISK

        return {
            "patient_id": profile.patient_id,
            "mfs_score": mfs,
            "mfs_risk": "high" if mfs >= cutoff else ("medium" if mfs >= Thresholds.MFS_MEDIUM_RISK else "low"),
            "frid_score": round(frid, 1),
            "frid_cat1_count": profile.frid_cat1_count,
            "frid_cat2_count": profile.frid_cat2_count,
            "baseline_age_multiplier": compute_age_multiplier(profile.age),
            "baseline_condition_multiplier": compute_condition_multiplier(profile),
            "baseline_bmi_multiplier": compute_bmi_multiplier(profile.bmi),
            "bmi": round(profile.bmi, 1),
            "ward_type": profile.ward_type,
            "status": "admitted"
        }

    def process_reading(self, patient_id: str, reading: IMUReading) -> dict:
        """Process one IMU reading for a patient. Called per sensor tick."""
        if patient_id not in self._engines:
            return {"error": f"Patient {patient_id} not admitted"}

        result = self._engines[patient_id].process(reading)
        self._gait[patient_id].push(reading)
        gait = self._gait[patient_id]

        return {
            "patient_id": patient_id,
            "timestamp": result.timestamp,
            "composite_score": result.composite_score,
            "risk_level": result.risk_level,
            "postural_state": result.postural_state.value,
            "motion_event": result.motion_event.value,
            "fall_category": result.fall_category.value,
            "sdi": round(result.sdi, 2),
            "alerts": result.alerts,
            "dominant_trigger": result.dominant_trigger,
            "score_breakdown": {
                "mfs_score": result.mfs_score,
                "frid_score": round(result.frid_score, 1),
                "dynamic_score": round(result.dynamic_score, 1),
                "age_multiplier": result.age_multiplier,
                "bmi_multiplier": result.bmi_multiplier,
                "condition_multiplier": result.condition_multiplier,
            },
            "gait_metrics": {
                "step_count": gait.step_count,
                "gait_speed_ms": gait.gait_speed_ms,
                "last_tug_time_s": gait.last_tug_time,
                "tug_risk_level": gait.tug_risk_level,
            }
        }

    def get_ward_overview(self) -> list:
        """Return risk summary for all admitted patients — for dashboard ward panel."""
        overview = []
        for pid, engine in self._engines.items():
            last = engine._last_result
            if last:
                overview.append({
                    "patient_id": pid,
                    "name": f"{engine.profile.name[0]}*** {engine.profile.name.split()[-1][0]}***" if " " in engine.profile.name else f"{engine.profile.name[:2]}***", # Data Privacy: Pseudonymization
                    "age": engine.profile.age,
                    "ward_type": engine.profile.ward_type,
                    "composite_score": last.composite_score,
                    "risk_level": last.risk_level,
                    "postural_state": last.postural_state.value,
                    "dominant_trigger": last.dominant_trigger,
                    "mfs_score": last.mfs_score,
                })
        return sorted(overview, key=lambda x: x["composite_score"], reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 16: FASTAPI INTEGRATION — drop-in replacement for existing backend
# ─────────────────────────────────────────────────────────────────────────────

# Instantiate the single ward manager — this replaces patient_history dict
ward = WardManager()

@app.get("/api/patients", response_model=List[Patient])
def get_patients(db: Session = Depends(get_db)):
    return crud.get_all_patients(db)

@app.post("/api/register")
def register_patient(patient_in: PatientRegister, db: Session = Depends(get_db)):
    import uuid
    patient_id = f"PT-{str(uuid.uuid4())[:4].upper()}"
    crud.register_patient(db, patient_in, patient_id)
    return {"status": "success", "patient_id": patient_id}

@app.post("/api/assign")
def assign_patient(patient_in: PatientAssign, db: Session = Depends(get_db)):
    db_patient = crud.get_patient(db, patient_in.patient_id)
    if not db_patient:
        raise HTTPException(status_code=404, detail="Patient not found")
        
    profile = PatientProfile(
        patient_id=patient_in.patient_id,
        name=db_patient.name,
        age=db_patient.age,
        weight_kg=db_patient.weight_kg or 70.0,
        height_cm=db_patient.height_cm or 170.0,
        parkinsons=patient_in.parkinsons,
        post_stroke=patient_in.post_stroke,
        peripheral_neuropathy=patient_in.muscle_issues,
        vision_impaired=patient_in.vision_issues,
        on_opioids=patient_in.on_opioids,
        on_antipsychotics=patient_in.on_antipsychotics,
        on_antidepressants=patient_in.on_antidepressants,
        on_benzodiazepines=patient_in.on_benzodiazepines,
        on_antiparkinson=patient_in.on_antiparkinson,
        on_antiepileptics=patient_in.on_antiepileptics,
        on_antihypertensives=patient_in.on_antihypertensives,
        on_diuretics=patient_in.on_diuretics,
        on_beta_blockers=patient_in.on_beta_blockers,
        on_alpha_blockers=patient_in.on_alpha_blockers,
        on_nsaids=patient_in.on_nsaids,
        on_muscle_relaxants=patient_in.on_muscle_relaxants,
        ward_type=patient_in.ward_type
    )
    result = ward.admit_patient(profile)
    
    crud.assign_patient_band(db, patient_in, float(result["mfs_score"]))
    return {"status": "success", "baseline_risk": result["mfs_score"], "assigned_band": patient_in.assigned_band, "ward_result": result}

@app.post("/api/sensor/vitals")
def receive_vitals(reading: SpO2Reading, db: Session = Depends(get_db)):
    patient_id = crud.update_db_vitals(db, reading.device_id, reading.heart_rate, reading.spo2)
    if not patient_id:
        return {"status": "ignored", "error": "No active patient mapped to this SpO2 sensor"}
    return {"status": "success", "patient_id": patient_id}

@app.post("/api/band-data")
def receive_band_data(reading: BandReading, db: Session = Depends(get_db)):
    patient = crud.get_patient(db, reading.patient_id)
    if not patient:
        return {"error": "Patient not found. Has the nurse admitted them?"}

    # -- Auto-Calibration for the Sway Map (UI Visuals) --
    if not patient.band_data.get("calibrated", False):
        offset_x = reading.grav_x
        offset_z = reading.grav_z
        patient.band_data["calibrated"] = True
        patient.band_data["offset_x"] = offset_x
        patient.band_data["offset_z"] = offset_z
    else:
        offset_x = patient.band_data.get("offset_x", 0.0)
        offset_z = patient.band_data.get("offset_z", 0.0)

    calib_grav_x = reading.grav_x - offset_x
    calib_grav_z = reading.grav_z - offset_z

    v_data = patient.vitals or {}
    real_spo2 = v_data.get("spo2", reading.spo2)
    real_hr = v_data.get("hr", reading.heart_rate)

    # 1. Convert to new ClinicalGuard Algorithm Reading
    imu = IMUReading(
        timestamp=time.time(),
        acc_x=reading.accel_x,
        acc_y=reading.accel_y,
        acc_z=reading.accel_z,
        gyro_x=reading.gyro_x,
        gyro_y=reading.gyro_y,
        gyro_z=reading.gyro_z,
        spo2=real_spo2,
        heart_rate=real_hr,
    )
    
    # 2. Process via ClinicalGuard Algorithm
    result = ward.process_reading(reading.patient_id, imu)
    
    # Failsafe: if patient wasn't admitted in WardManager memory (e.g. server reset):
    if "error" in result:
        meds = patient.medications or []
        profile = PatientProfile(
            patient_id=patient.id, name=patient.name, age=patient.age,
            weight_kg=patient.weight_kg or 70.0, height_cm=patient.height_cm or 170.0, 
            parkinsons=patient.parkinsons,
            peripheral_neuropathy=patient.muscle_issues,
            vision_impaired=patient.vision_issues,
            on_opioids="on_opioids" in meds,
            on_antipsychotics="on_antipsychotics" in meds,
            on_antidepressants="on_antidepressants" in meds,
            on_benzodiazepines="on_benzodiazepines" in meds,
            on_antiparkinson="on_antiparkinson" in meds,
            on_antiepileptics="on_antiepileptics" in meds,
            on_antihypertensives="on_antihypertensives" in meds,
            on_diuretics="on_diuretics" in meds,
            on_beta_blockers="on_beta_blockers" in meds,
            on_alpha_blockers="on_alpha_blockers" in meds,
            on_nsaids="on_nsaids" in meds,
            on_muscle_relaxants="on_muscle_relaxants" in meds
        )
        ward.admit_patient(profile)
        # Restore prior gait step count from db
        ward._gait[patient.id]._total_steps = patient.band_data.get("step_count", 0) if patient.band_data else 0
        result = ward.process_reading(reading.patient_id, imu)

    # 3. Synchronize with SQLite Database
    vitals_updates = {"spo2": real_spo2, "hr": real_hr}
    gait = result.get("gait_metrics", {})
    score_bd = result.get("score_breakdown", {})
    band_updates = {
        "calibrated": True,
        "offset_x": offset_x,
        "offset_z": offset_z,
        "grav_x": round(calib_grav_x, 2),
        "grav_z": round(calib_grav_z, 2),
        "last_svm": round(math.sqrt(reading.accel_x**2 + reading.accel_y**2 + reading.accel_z**2) / 9.81, 2),
        "sdi": round(result["sdi"], 2),
        "motion_event": result["motion_event"],
        "postural_state": result["postural_state"],
        "fall_category": result.get("fall_category", "accidental"),
        "dominant_trigger": result.get("dominant_trigger", "stable"),
        "risk_level": result.get("risk_level", "LOW"),
        # Score breakdown for dashboard cards
        "mfs_score": score_bd.get("mfs_score", 0),
        "frid_score": score_bd.get("frid_score", 0.0),
        "dynamic_score": score_bd.get("dynamic_score", 0.0),
        "age_multiplier": score_bd.get("age_multiplier", 1.0),
        "bmi_multiplier": score_bd.get("bmi_multiplier", 1.0),
        "condition_multiplier": score_bd.get("condition_multiplier", 1.0),
        # Gait metrics
        "step_count": gait.get("step_count", 0),
        "gait_speed_ms": gait.get("gait_speed_ms"),
        "tug_risk_level": gait.get("tug_risk_level", "not_measured"),
    }
    
    # Implement medical-grade database throttler (2.0s) to prevent SQLite locks during high-frequency streams
    now = time.time()
    last_save = getattr(app, '_db_last_saves', {})
    if not hasattr(app, '_db_last_saves'):
        app._db_last_saves = last_save
        
    last_t = last_save.get(reading.patient_id, 0.0)
    
    # Crucial fall events bypass the throttle to guarantee immediate alert recording
    critical_event = band_updates.get("risk_level") in ["HIGH", "CRITICAL"] or result.get("alerts")
    
    if critical_event or (now - last_t > 2.0):
        crud.update_live_risk(
            db, 
            reading.patient_id, 
            float(result["composite_score"]), 
            result["alerts"], 
            band_updates, 
            vitals_updates
        )
        last_save[reading.patient_id] = now
    
    return {"status": "success", "risk": result["composite_score"], "svm": band_updates["last_svm"], "sdi": result["sdi"]}

@app.get("/api/ward")
def get_ward():
    return ward.get_ward_overview()




# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST — runs when executed directly
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random

    print("=" * 60)
    print("ClinicalGuard Algorithm — Integration Test")
    print("=" * 60)

    # Admit a Parkinson's patient
    pd_patient = PatientProfile(
        patient_id="PT001", name="Anwar M.", age=72,
        weight_kg=68, height_cm=172,
        parkinsons=True, peripheral_neuropathy=True,
        on_antiparkinson=True, on_antidepressants=True,
        prior_fall_history=True, fall_count_last_3m=2,
        tug_time_s=16.0, uses_ambulatory_aid="walker",
        iv_line=True, mental_status="oriented",
        ward_type="neuro"
    )
    result = ward.admit_patient(pd_patient)
    print(f"\nAdmit PT001 (Parkinson's):")
    print(f"  MFS score:            {result['mfs_score']} ({'HIGH' if result['mfs_score']>=45 else 'MEDIUM'})")
    print(f"  FRID score:           {result['frid_score']}")
    print(f"  Age multiplier:       {result['baseline_age_multiplier']}x")
    print(f"  Condition multiplier: {result['baseline_condition_multiplier']}x")

    # Simulate 35 readings — calibration + normal + fall trajectory
    base_t = time.time()
    for i in range(35):
        t = base_t + i * 0.5
        # After reading 20: simulate fall
        if i < 20:
            ax, ay, az = random.gauss(-3.5, 0.3), random.gauss(0.5, 0.2), random.gauss(9.3, 0.3)
            gx, gy, gz = random.gauss(0.1, 0.05), random.gauss(0.05, 0.02), random.gauss(0.0, 0.03)
        else:
            # Rapid tilt — fall trajectory
            factor = min((i - 20) * 0.3, 1.0)
            ax = random.gauss(-3.5 + factor * 14, 0.5)
            ay = random.gauss(0.5 + factor * 8, 0.5)
            az = random.gauss(9.3 - factor * 4, 0.5)
            gx = random.gauss(0.1 + factor * 0.8, 0.1)
            gy = random.gauss(0.05 + factor * 0.6, 0.1)
            gz = random.gauss(0.0, 0.05)

        reading = IMUReading(
            timestamp=t, acc_x=ax, acc_y=ay, acc_z=az,
            gyro_x=gx, gyro_y=gy, gyro_z=gz,
            spo2=97 - (i > 25) * 6,
            heart_rate=72
        )
        out = ward.process_reading("PT001", reading)
        if i in [19, 24, 29, 34] or out.get("risk_level") in ("HIGH", "CRITICAL"):
            print(f"\n  Reading {i+1}: t={i*0.5:.1f}s")
            print(f"    Score: {out['composite_score']}  Level: {out['risk_level']}")
            print(f"    State: {out['postural_state']}  Motion: {out['motion_event']}")
            print(f"    SDI:   {out['sdi']}  Trigger: {out['dominant_trigger']}")
            if out['alerts']:
                print(f"    Alert: {out['alerts'][0]}")

    print("\n" + "=" * 60)
    print("Algorithm test complete.")