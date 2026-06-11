import pandas as pd
from pathlib import Path

OUT_DIR = Path.home() / "argentina-rct/output"
master  = pd.read_csv(OUT_DIR / "master.csv")

dictionary = {
    # ── IDENTIFIERS ──────────────────────────────────────────
    "id_SSE":               "Individual social security ID (spine of all merges)",
    "id_correo":            "Email-based ID (links to portal/SkillLab data)",

    # ── TREATMENT ────────────────────────────────────────────
    "treat":                "Treatment assignment: 0=Control, 1=T1_CV, 2=T2_Full, 3=T3_InDemand",
    "treat_label":          "Treatment label (string version of treat)",
    "treat_phase":          "Phase of treatment rollout",
    "treat_any":            "1 if assigned to any treatment arm",
    "invited":              "1 if individual received SkillLab invitation email",
    "registered":           "1 if individual registered on SkillLab platform (take-up)",

    # ── STRATA / GEOGRAPHY ───────────────────────────────────
    "provincia":            "Province of residence",
    "agenciaterritorial":   "Territorial employment agency",
    "municipio":            "Municipality name",
    "muni_id":              "Municipality ID (used as strata FE in regressions)",
    "región":               "Region of Argentina",

    # ── DEMOGRAPHICS ─────────────────────────────────────────
    "edad":                 "Age in years",
    "edad_above_med":       "1 if age is above sample median (strata variable)",
    "sexo":                 "Sex (raw)",
    "mujer":                "1 if female",
    "genero_autopercibido": "Self-identified gender",
    "pueblo_originario":    "1 if indigenous",
    "etnia":                "Ethnicity",
    "discapacidad":         "1 if has a disability",
    "fecha_nacimiento":     "Date of birth",

    # ── SOCIOECONOMIC ─────────────────────────────────────────
    "nivel_educativo":      "Education level (raw string)",
    "id_nivel_educativo":   "Education level ID",
    "educ_primary":         "1 if highest education is primary",
    "educ_secondary":       "1 if completed secondary education",
    "te_gustaria_finalizar_primaria":   "1 if wants to finish primary school",
    "te_gustaria_finalizar_secundaria": "1 if wants to finish secondary school",
    "work_experience":      "1 if has prior work experience",
    "experiencia_laboral_anterior": "Prior work experience (raw)",
    "has_children":         "1 if has children under 18",
    "hijos_menores":        "Number of minor children",
    "oficios_capactacion":  "Trade/vocational training interest",
    "caracterizacion":      "Beneficiary characterization category",
    "trabaja_actualmente":  "1 if currently working at registration",

    # ── PROGRAM ──────────────────────────────────────────────
    "fomentar":             "1 if enrolled in Fomentar Empleo (0 = Volver al Trabajo)",
    "grupo":                "Program group",
    "estado":               "Beneficiary status in program",
    "acepta_gestion_oe":    "1 if accepts employment office management",
    "población2022indec":   "Municipality population (2022 INDEC census)",
    "cantidadvataoctubre2024":      "Number of VAT beneficiaries in municipality (Oct 2024)",
    "cantidadfomentarft1yft2a":     "Number of Fomentar beneficiaries in municipality",
    "merge_OE_data":        "Merge flag: employment office data",
    "merge_OE_data_x":      "Merge flag duplicate (from crosswalk merge)",
    "merge_participant_data": "Merge flag: participant data",

    # ── ADMIN OUTCOMES ───────────────────────────────────────
    "receiving_benefit":         "1 if currently receiving a government social benefit",
    "enrolled_in_course":        "1 if enrolled in a vocational training course (portal)",
    "applied_to_portal_job":     "1 if applied to a job via the employment portal",
    "allowed_companies_contact": "1 if gave permission for companies to contact them",
    "updated_cv":                "1 if updated their CV on the portal",
    "emp_latest":                "1 if formally employed in most recent available month",
    "emp_latest_month":          "Most recent month with employment record (YYYY-MM)",

    # ── SURVEY OUTCOMES (midline, ~3 months post-treatment) ──
    "P0":   "Days worked last month (survey)",
    "P1":   "Number of portal visits in last 2 weeks (survey)",
    "P2":   "Hours spent job searching per week (survey)",
    "P3":   "Number of jobs applied to last month (survey)",
    "P4":   "1 if discovered a new skill through SkillLab (survey)",
    "P6":   "1 if considered a new career through SkillLab (survey)",
    "P8":   "Reservation wage in ARS — minimum acceptable salary (survey)",
    "P9":   "Job-finding confidence: 1=very unlikely, 10=very likely (survey)",
    "P10":  "1 if received a job offer in last 3 months (survey)",
    "P11":  "1 if used SkillLab platform (take-up, survey)",
    "P12":  "SkillLab satisfaction rating 1-10 (survey, conditional on use)",
}

# ── PRINT ────────────────────────────────────────────────────
print(f"Master dataset: {master.shape[0]:,} rows x {master.shape[1]} cols")
print(f"Columns in master not in dictionary: "
      f"{[c for c in master.columns if c not in dictionary]}")
print()

rows = []
for col, desc in dictionary.items():
    if col in master.columns:
        dtype    = str(master[col].dtype)
        n_miss   = master[col].isna().sum()
        pct_miss = n_miss / len(master)
        if master[col].dtype in ["float64", "int64"]:
            summary = f"mean={master[col].mean():.3f}, min={master[col].min():.0f}, max={master[col].max():.0f}"
        else:
            top = master[col].value_counts().index[0] if master[col].notna().any() else "N/A"
            summary = f"top='{top}'"
        rows.append({
            "column":      col,
            "description": desc,
            "dtype":       dtype,
            "missing":     n_miss,
            "pct_missing": f"{pct_miss:.1%}",
            "summary":     summary,
        })

df_dict = pd.DataFrame(rows)
print(df_dict.to_string(index=False))

# ── SAVE ─────────────────────────────────────────────────────
df_dict.to_csv(OUT_DIR / "data_dictionary.csv", index=False)
print(f"\nSaved: {OUT_DIR / 'data_dictionary.csv'}")
