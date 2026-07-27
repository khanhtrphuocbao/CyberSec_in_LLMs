"""
Example Medical Text Data for RAG Ingestion
==========================================
This script creates sample medical documents for testing the RAG module.

In production, replace with actual medical textbooks, clinical guidelines,
or MedlinePlus content downloaded from https://medlineplus.gov/

Run this script to create sample data, then use RAG.py to ingest it.
"""

import os
from pathlib import Path

# Create sample medical texts directory
SAMPLE_DIR = Path("./medqa_rag/sample_medical_texts")
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_TEXTS = {
    "cardiology_basics.txt": """
CARDIOLOGY BASICS - Medical Textbook Chapter

HEART FAILURE
Heart failure (HF) is a clinical syndrome characterized by structural or functional impairment of ventricular filling or ejection of blood. It affects approximately 6.5 million Americans and is a leading cause of morbidity and mortality.

TYPES OF HEART FAILURE:
1. Systolic Heart Failure (HFrEF): Reduced ejection fraction (<40%). The heart cannot contract effectively.
2. Diastolic Heart Failure (HFpEF): Preserved ejection fraction. The heart cannot relax properly.

LEFT-SIDED VS RIGHT-SIDED HEART FAILURE:
- Left-sided failure: Pulmonary congestion, dyspnea, orthopnea, PND
- Right-sided failure: Peripheral edema, JVD, hepatomegaly, ascites

SYMPTOMS:
- Dyspnea (shortness of breath)
- Fatigue and exercise intolerance
- Peripheral edema
- Orthopnea (shortness of breath when lying flat)
- Paroxysmal nocturnal dyspnea (PND)

SIGNS:
- S3 gallop (indicates volume overload)
- Pulmonary crackles
- Jugular venous distension (JVD)
- Hepatomegaly
- Peripheral edema

DIAGNOSTIC TESTS:
1. Echocardiography: Assess EF, wall motion, valvular function
2. BNP/NT-proBNP: Elevated in heart failure
3. Chest X-ray: Cardiomegaly, pulmonary congestion
4. ECG: May show arrhythmias, LVH, ischemia

TREATMENT:
- ACE inhibitors (reduce mortality)
- Beta-blockers (carvedilol, metoprolol succinate, bisoprolol)
- Aldosterone antagonists (spironolactone)
- Diuretics (symptom relief)
- ARBs if ACE-I intolerant
- Hydralazine + Nitrate in African Americans
- Device therapy: ICD, CRT for select patients
""",

    "ace_inhibitors.txt": """
ACE INHIBITORS - Pharmacology

MECHANISM OF ACTION:
ACE inhibitors block the angiotensin-converting enzyme, which converts angiotensin I to angiotensin II. Angiotensin II is a potent vasoconstrictor and stimulates aldosterone release, leading to sodium and water retention.

By inhibiting ACE:
1. Decreased angiotensin II → vasodilation
2. Decreased aldosterone → reduced sodium/water retention
3. Increased bradykinin (beneficial vasodilator)

COMMON ACE INHIBITORS:
- Lisinopril
- Enalapril
- Ramipril
- Captopril
- Benazepril

INDICATIONS:
- Hypertension
- Heart failure
- Post-MI with reduced EF
- Diabetic nephropathy
- Chronic kidney disease (proteinuria)

CONTRAINDICATIONS:
- Pregnancy (teratogenic - fetal harm)
- Bilateral renal artery stenosis
- History of angioedema
- Hyperkalemia

SIDE EFFECTS:
1. Dry cough (due to increased bradykinin) - 5-20% of patients
2. Angioedema (rare but serious)
3. Hyperkalemia
4. Hypotension (especially first-dose)
5. Acute kidney injury (in renal artery stenosis)

LABORATORY MONITORING:
- Serum potassium (watch for hyperkalemia)
- Serum creatinine (watch for AKI)
- CBC (watch for neutropenia with captopril)

CLINICAL Pearl:
First-dose hypotension is common. Start with low doses, especially in:
- Elderly patients
- Those on high-dose diuretics
- Patients with heart failure
""",

    "diabetes_management.txt": """
DIABETES MELLITUS - Type 2 Diabetes

DEFINITION:
Type 2 diabetes mellitus (T2DM) is a metabolic disorder characterized by:
- Insulin resistance
- Relative insulin deficiency
- Hyperglycemia

DIAGNOSTIC CRITERIA (ADA Guidelines):
1. Fasting plasma glucose ≥126 mg/dL (7.0 mmol/L)
2. 2-hour plasma glucose ≥200 mg/dL during OGTT
3. HbA1c ≥6.5%
4. Random plasma glucose ≥200 mg/dL with classic symptoms

PRE-DIABETES:
- Impaired fasting glucose: 100-125 mg/dL
- Impaired glucose tolerance: 140-199 mg/dL
- HbA1c: 5.7-6.4%

COMPLICATIONS:
MICROVASCULAR:
- Retinopathy
- Nephropathy
- Neuropathy

MACROVASCULAR:
- Coronary artery disease
- Cerebrovascular disease
- Peripheral vascular disease

TREATMENT APPROACH:

First-line: Metformin + Lifestyle modification

Second-line options (if not at goal after 3 months):
- Sulfonylureas (stimulate insulin secretion)
- DPP-4 inhibitors
- GLP-1 receptor agonists
- SGLT2 inhibitors (also cardioprotective)
- Thiazolidinediones

Third-line: Insulin therapy

SPECIAL CONSIDERATIONS:
- Metformin contraindicated in severe renal impairment (eGFR <30)
- SGLT2 inhibitors: Associated with euglycemic ketoacidosis risk
- GLP-1 agonists: GI side effects, NOT for patients with gastroparesis

TARGETS:
- HbA1c: <7% for most adults
- Preprandial glucose: 80-130 mg/dL
- Postprandial glucose: <180 mg/dL
""",

    "renal_physiology.txt": """
RENAL PHYSIOLOGY AND ACUTE KIDNEY INJURY

RENAL FUNCTION OVERVIEW:
The kidneys filter approximately 180L of blood daily, producing 1-2L of urine. They regulate:
- Fluid and electrolyte balance
- Acid-base balance
- Blood pressure (via RAAS)
- Erythropoiesis (via EPO)
- Vitamin D activation

GLOMERULAR FILTRATION RATE (GFR):
Normal GFR: 90-120 mL/min/1.73m²

CKD Stages:
- Stage 1: GFR ≥90 (with kidney damage)
- Stage 2: GFR 60-89
- Stage 3a: GFR 45-59
- Stage 3b: GFR 30-44
- Stage 4: GFR 15-29
- Stage 5: GFR <15 or on dialysis

ACUTE KIDNEY INJURY (AKI):
Definition (KDIGO criteria):
- Increase in creatinine by ≥0.3 mg/dL within 48 hours
- OR increase in creatinine to ≥1.5× baseline within 7 days
- OR urine output <0.5 mL/kg/h for 6 hours

CAUSES - "HRSCTOP":
- Hypovolemia
- Renal hypoperfusion (prerenal)
- Sepsis
- Cyclosporine/NSAIDs/ACEi
- Toxins (contrast, aminoglycosides)
- Obstruction (postrenal)
- Glomerulonephritis (intrinsic)

PRERENAL AKI:
- Caused by decreased renal perfusion
- Responds to fluid resuscitation
- BUN:Cr ratio >20:1
- FENa <1%
- Urine Na <20 mEq/L

INTRINSIC AKI:
- Acute tubular necrosis (ATN) - most common
- Acute interstitial nephritis (AIN)
- Glomerulonephritis
- Vasculitis

POSTRENAL AKI:
- Bilateral obstruction
- Obstruction of solitary kidney
""",

    "infectious_disease.txt": """
INFECTIOUS DISEASES - Antibiotic Therapy

ANTIBIOTIC CLASSES AND MECHANISMS:

BETA-LACTAMS:
Mechanism: Inhibit cell wall synthesis

1. Penicillins:
   - Ampicillin/Amoxicillin: Wide spectrum, susceptible to beta-lactamases
   - Piperacillin: Antipseudomonal coverage
   - Nafcillin/Oxacillin: MRSA (if susceptible)

2. Cephalosporins:
   - 1st gen (cefazolin): Gram-positive cocci, some gram-negatives
   - 2nd gen (cefuroxime): Added H. influenzae, anaerobes
   - 3rd gen (ceftriaxone): Wide gram-negative coverage, crosses BBB
   - 4th gen (cefepime): Pseudomonas coverage
   - 5th gen (ceftaroline): MRSA coverage

3. Carbapenems:
   - Meropenem, Imipenem, Ertapenem
   - Broadest spectrum beta-lactams
   - Reserved for ESBL infections, nosocomial infections

AMINOGLYCOSIDES:
- Gentamicin, Tobramycin, Amikacin
- Mechanism: Inhibit 30S ribosomal subunit (protein synthesis)
- Only for serious gram-negative infections
- Monitor: Serum levels, renal function, ototoxicity

FLUOROQUINOLONES:
- Ciprofloxacin, Levofloxacin, Moxifloxacin
- Mechanism: Inhibit DNA gyrase and topoisomerase IV
- Broad spectrum, including atypicals
- BLACK BOX WARNING: Tendon rupture, QT prolongation

CLINDAMYCIN:
- 50S ribosomal inhibitor
- Good for anaerobes, GAS, staph
- Associated with C. difficile colitis

VANCOMYCIN:
- Cell wall synthesis inhibitor
- Reserved for MRSA, serious gram-positive infections
- Monitor: Trough levels (15-20 for serious infections)
"""
}


def create_sample_texts():
    """Create sample medical text files for testing."""
    print("[Setup] Creating sample medical texts...")

    for filename, content in SAMPLE_TEXTS.items():
        filepath = SAMPLE_DIR / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content.strip())
        print(f"  Created: {filepath}")

    print(f"\n[Setup] Created {len(SAMPLE_TEXTS)} sample medical texts in {SAMPLE_DIR}")
    print("\nTo ingest these into the RAG system:")
    print(f"  from RAG import MedQA_RAG")
    print(f"  rag = MedQA_RAG(api_key='your-key')")
    print(f"  rag.ingest_documents('{SAMPLE_DIR}')")


if __name__ == "__main__":
    create_sample_texts()
