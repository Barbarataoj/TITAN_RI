✅ README: Phase I OUD–HIV–Overdose Feature Specification

This document outlines the Phase I feature set for extending TITAN to simulate:

People with Opioid Use Disorder (OUD)

HIV prevention and treatment in OUD networks

Overdose incidence & overdose deaths

Integration with existing TITAN sexual and injection network components

This is the foundation before adding HCV and more complex OUD dynamics in Phase II.

1. Goals of Phase I

Phase I aims to build a minimal but complete modeling framework to evaluate:

MOUD expansion

HIV testing scale-up

HIV prevention (PrEP)

HIV treatment (HAART)

SSP/harm reduction

Overdose and overdose death

These interventions interact with the OUD population through TITAN’s network-based structure.

Phase I does NOT include HCV (that will be Phase II).

2. Existing TITAN Features Used in Phase I
Feature	Purpose	Notes
base_feature	Core agent lifecycle & network	No change needed
high_risk	Behavioral risk flag	Used by MOUD, testing, overdose
incar	Incarceration dynamics	Optional interaction later
prep	HIV PrEP	Already works; interacts with HIV testing
haart	HIV treatment	Will use hiv_dx from HIV testing
partner_tracing	Notify partners for testing	Works automatically once testing is added
syringe_services	SSP / harm reduction	Lowers injection & overdose risk
vaccine	Monkeypox (optional)	Not essential for OUD modeling
random_trial	Utility	Used everywhere

These features remain unchanged but are consumed by the new features.

3. New Features to Add in Phase I
3.1 MOUD (moud.py) — Medication for Opioid Use Disorder

Purpose

Represent MOUD enrollment, adherence, and dropout.

Agent state

agent.moud

Effects

↓ Injection risk

↓ Overdose risk

YAML parameters

moud:
  init: 0.2
  enroll:
    begin:
      prob: 0.02
      start: 12
      stop: 36
    end:
      prob: 0.05
      start: 36
      stop: 120
  reinit:
    prob: 0.02
  adherence:
    init: 0.8
    prob: 0.8
  discontinue: 0.05
  high_risk_mult: 1.5

3.2 HIV Testing (hiv_testing.py)

Purpose

Convert undiagnosed HIV to diagnosed HIV.

Feed into HAART and partner tracing.

Agent state

agent.hiv_dx

YAML parameters

hiv_testing:
  base_prob: 0.05
  high_risk_mult: 2.0
  oud_mult: 1.5
  interval: 3


Effects

Diagnosed positives → HAART (existing feature)

Triggers partner tracing

3.3 Overdose (overdose.py)

Purpose

Simulate overdose incidence and overdose deaths among OUD.

Agent state

agent.overdose_count

Optional: agent.last_overdose_time

Fatal overdose → exit_class = "death"

YAML parameters

overdose:
  base_prob_inj: 0.01
  base_prob_noninj: 0.003
  fatal_frac: 0.15
  moud_rr: 0.4
  high_risk_mult: 2.0
  ssp_rr: 0.8


Effects

Increases deaths from overdose

Reduced by MOUD and SSP

4. Where these new features plug into TITAN

All new features should be placed in:

titan/features/


Specifically:

titan/features/moud.py
titan/features/hiv_testing.py
titan/features/overdose.py


After adding the files, register them in:

titan/features/__init__.py


Example:

from .moud import MOUDFeature
from .hiv_testing import HIVTestingFeature
from .overdose import OverdoseFeature

FEATURES = {
    ...
    "moud": MOUDFeature,
    "hiv_testing": HIVTestingFeature,
    "overdose": OverdoseFeature,
}


Then enable them in YAML:

features:
  moud: true
  hiv_testing: true
  overdose: true

5. Summary of Phase I Architecture
What the model will support after Phase I

✔ OUD population structure (Inj, NonInj, None)
✔ Injection & sexual networks
✔ HIV transmission
✔ PrEP
✔ HAART
✔ HIV testing (new)
✔ MOUD (new)
✔ SSP
✔ Overdose incidence & deaths (new)

What the model will NOT support yet

❌ HCV transmission
❌ HCV testing & treatment
❌ Dynamic OUD transitions (initiation/cessation/relapse)
❌ Detailed incarceration release overdose surge
❌ Multi-disease synergy

Those are Phase II–III.