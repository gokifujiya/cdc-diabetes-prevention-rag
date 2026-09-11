from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "pcd_type2_diabetes_metadata.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "pcd_type2_diabetes_screened.csv"


# ---------------------------------------------------------
# Terms strongly related to prevention of type 2 diabetes
# ---------------------------------------------------------

STRONG_PREVENTION_TERMS = {
    "diabetes prevention program": 5,
    "national diabetes prevention program": 5,
    "prevent type 2 diabetes": 5,
    "preventing type 2 diabetes": 5,
    "prevention of type 2 diabetes": 5,
    "prevent or delay": 5,
    "prediabetes": 4,
    "prediabetic state": 4,
    "diabetes prevention": 4,
    "lifestyle intervention": 4,
    "lifestyle change program": 4,
    "dpp": 3,
}

MODERATE_PREVENTION_TERMS = {
    "risk reduction": 2,
    "risk factor": 1,
    "weight loss": 2,
    "physical activity": 2,
    "healthy eating": 2,
    "diet": 1,
    "obesity": 1,
    "overweight": 1,
    "screening": 2,
    "referral": 2,
    "health promotion": 2,
    "behavioral intervention": 2,
    "community intervention": 2,
    "worksite intervention": 2,
    "lifestyle": 1,
    "prevention strategies": 3,
}


# ---------------------------------------------------------
# Terms suggesting treatment/management of established T2D
# rather than prevention of developing T2D
# ---------------------------------------------------------

ESTABLISHED_DIABETES_TERMS = {
    "diabetes management": 3,
    "self-management": 3,
    "glycemic control": 3,
    "glycaemic control": 3,
    "diabetes control": 3,
    "diabetes care": 2,
    "medication adherence": 3,
    "antihyperglycemic": 3,
    "insulin therapy": 3,
    "diabetic retinopathy": 4,
    "diabetic neuropathy": 4,
    "diabetic nephropathy": 4,
    "chronic kidney disease": 2,
    "foot care": 3,
    "diabetes complications": 4,
    "hba1c target": 2,
}


def normalize_text(value):
    if pd.isna(value):
        return ""
    return str(value).lower()


def build_search_text(row):
    fields = [
        row.get("title", ""),
        row.get("metadata_title", ""),
        row.get("abstract", ""),
        row.get("subjects", ""),
        row.get("keywords", ""),
    ]

    return " ".join(normalize_text(value) for value in fields)


def find_terms(text, term_dictionary):
    matches = []

    for term, weight in term_dictionary.items():
        if term in text:
            matches.append((term, weight))

    return matches


def screen_record(row):
    text = build_search_text(row)

    strong_matches = find_terms(text, STRONG_PREVENTION_TERMS)
    moderate_matches = find_terms(text, MODERATE_PREVENTION_TERMS)
    exclusion_matches = find_terms(text, ESTABLISHED_DIABETES_TERMS)

    prevention_score = sum(weight for _, weight in strong_matches)
    prevention_score += sum(weight for _, weight in moderate_matches)

    treatment_score = sum(weight for _, weight in exclusion_matches)

    strong_terms_found = [term for term, _ in strong_matches]
    moderate_terms_found = [term for term, _ in moderate_matches]
    exclusion_terms_found = [term for term, _ in exclusion_matches]

    # -----------------------------------------------------
    # Screening rules
    # -----------------------------------------------------

    # Clearly prevention-oriented
    if prevention_score >= 5:
        status = "Include"

    # Some prevention relevance, but manual review desirable
    elif prevention_score >= 2:
        status = "Maybe"

    # Mostly established-diabetes management/complications
    elif treatment_score >= 3:
        status = "Exclude"

    # Insufficient evidence from metadata alone
    else:
        status = "Maybe"

    reasons = []

    if strong_terms_found:
        reasons.append(
            "Strong prevention terms: "
            + ", ".join(strong_terms_found)
        )

    if moderate_terms_found:
        reasons.append(
            "Other prevention terms: "
            + ", ".join(moderate_terms_found)
        )

    if exclusion_terms_found:
        reasons.append(
            "Established-diabetes terms: "
            + ", ".join(exclusion_terms_found)
        )

    if not reasons:
        reasons.append(
            "No decisive prevention or treatment terms found."
        )

    return pd.Series(
        {
            "screening_status": status,
            "prevention_score": prevention_score,
            "treatment_score": treatment_score,
            "screening_reason": " | ".join(reasons),
        }
    )


# ---------------------------------------------------------
# Load metadata
# ---------------------------------------------------------

df = pd.read_csv(
    INPUT_FILE,
    dtype=str,
    encoding="utf-8-sig",
)


# ---------------------------------------------------------
# Screen all 74 candidate records
# ---------------------------------------------------------

screening_results = df.apply(
    screen_record,
    axis=1,
)

result = pd.concat(
    [df, screening_results],
    axis=1,
)


# ---------------------------------------------------------
# Sort so Include papers appear first
# ---------------------------------------------------------

status_order = {
    "Include": 0,
    "Maybe": 1,
    "Exclude": 2,
}

result["_status_order"] = result["screening_status"].map(status_order)

result = result.sort_values(
    by=["_status_order", "prevention_score"],
    ascending=[True, False],
)

result = result.drop(columns=["_status_order"])


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

result.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig",
)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print(f"Total records screened: {len(result)}")
print()

for status in ["Include", "Maybe", "Exclude"]:
    count = (result["screening_status"] == status).sum()
    print(f"{status}: {count}")

print()
print(f"Saved to: {OUTPUT_FILE}")
