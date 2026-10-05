# %% [markdown]
# # Student Academic Risk - Early Warning System
# Pipeline: 1) Explore  2) Preprocess  3) Define risk + model  4) Evaluate
#           5) Predict + recommend  6) Automated report
#
# Run:  python student_risk_pipeline.py [path_to_csv]

# %% Imports & configuration
import sys, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # headless: save figures to files
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
SEED = 42
# Paths are relative to this script's folder, so it runs anywhere (VS Code, terminal, etc.)
HERE = os.path.dirname(os.path.abspath(__file__))
CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "dcs_student_data.csv")
OUT = HERE
FIG = f"{OUT}/figures"
os.makedirs(FIG, exist_ok=True)

def save(name):
    plt.tight_layout(); plt.savefig(f"{FIG}/{name}.png", dpi=130); plt.close()

# %% ------------------------------------------------------------------
# 1. LOAD + DATA QUALITY AUDIT
# ---------------------------------------------------------------------
raw = pd.read_csv(CSV)
print(f"Loaded {raw.shape[0]} rows x {raw.shape[1]} columns")
raw.columns = raw.columns.str.strip()
raw = raw.rename(columns={"Attendance (%)": "Attendance"})   # simpler name for code

print("\n--- Data quality audit ---")
print("Missing values per column (only those with gaps):")
print(raw.isna().sum()[raw.isna().sum() > 0].to_string())
print("Duplicate Student_IDs:", raw.Student_ID.duplicated().sum())
print("Department labels (raw):", sorted(raw.Department.unique()))
print("Gender labels (raw):", sorted(raw.Gender.unique()))
print("Non-numeric math_score values:", raw.math_score[pd.to_numeric(raw.math_score, errors="coerce").isna()].tolist())

# %% ------------------------------------------------------------------
# 2. CLEANING
# ---------------------------------------------------------------------
df = raw.copy()

# 2a. Remove duplicate students (same Student_ID). Keep first occurrence.
df = df.drop_duplicates(subset="Student_ID", keep="first").reset_index(drop=True)

# 2b. Standardise text categories (case, whitespace, aliases such as CS == Computer Science)
df["Gender"] = df["Gender"].str.strip().str.title()
dept = df["Department"].str.strip().str.lower()
dept = dept.replace({"cs": "computer science", "math": "mathematics"})
df["Department"] = dept.str.title()

# 2c. math_score was read as text because of stray characters (e.g. '\t41') -> force numeric
df["math_score"] = pd.to_numeric(df["math_score"].astype(str).str.strip(), errors="coerce")

# 2d. Invalid values -> NaN (then imputed). Percent/score columns must be within 0-100,
#     participation within 0-10, age within a plausible student range.
pct_cols = ["Attendance", "Midterm_Score", "Final_Score", "Assignments_Avg", "Quizzes_Avg",
            "Projects_Score", "Total_Score", "math_score", "reading_score",
            "writing_score", "science_score"]
for c in pct_cols:
    df.loc[(df[c] < 0) | (df[c] > 100), c] = np.nan
df.loc[(df.Participation_Score < 0) | (df.Participation_Score > 10), "Participation_Score"] = np.nan
df.loc[(df.Age < 15) | (df.Age > 40), "Age"] = np.nan

# 2e. Impute: median for numeric (robust to outliers), mode for categorical.
#     Missing data is only ~2% per column, so imputation changes little.
num_cols = df.select_dtypes("number").columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
for c in ["Gender", "Department", "Grade"]:
    df[c] = df[c].fillna(df[c].mode()[0])

print(f"\nAfter cleaning: {df.shape[0]} students, {df.isna().sum().sum()} missing values left")
print("Departments:", df.Department.value_counts().to_dict())

# %% ------------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------------------
coursework = ["Midterm_Score", "Final_Score", "Assignments_Avg", "Quizzes_Avg", "Projects_Score"]
df["Participation_Pct"] = df["Participation_Score"] * 10          # put 0-10 on a 0-100 scale

# Performance index = weighted average of everything that is actually graded work.
# Weights mirror a typical university scheme (finals/midterms count most).
W = {"Midterm_Score": .25, "Final_Score": .30, "Assignments_Avg": .15,
     "Quizzes_Avg": .10, "Projects_Score": .15, "Participation_Pct": .05}
df["Performance_Index"] = sum(df[k] * w for k, w in W.items())

# --- Attendance vs marks -------------------------------------------------
print("\n--- Attendance vs marks ---")
rows = []
for target in ["Total_Score", "Performance_Index", "Midterm_Score", "Final_Score"]:
    r, p = stats.pearsonr(df.Attendance, df[target])
    rho, _ = stats.spearmanr(df.Attendance, df[target])
    rows.append((target, r, rho, p))
    print(f"Attendance vs {target:18s} Pearson r={r:+.3f}  Spearman={rho:+.3f}  p={p:.3f}")

# Sanity check on the dataset's own Total_Score / Grade columns
print("\nIs Total_Score consistent with the component scores?")
print("  corr(Total_Score, Performance_Index) =", round(df.Total_Score.corr(df.Performance_Index), 3))
print("Mean Total_Score by Grade (should rise A->F if Grade follows score):")
print(df.groupby("Grade").Total_Score.mean().round(1).to_string())

# --- Plots ---------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(df.Attendance, bins=30, kde=True, ax=ax[0]); ax[0].set_title("Attendance (%)")
sns.histplot(df.Total_Score, bins=30, kde=True, ax=ax[1], color="C1"); ax[1].set_title("Total_Score")
sns.histplot(df.Performance_Index, bins=30, kde=True, ax=ax[2], color="C2"); ax[2].set_title("Performance index")
save("01_distributions")

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for a, y in zip(ax, ["Total_Score", "Performance_Index"]):
    a.hexbin(df.Attendance, df[y], gridsize=35, cmap="Blues", mincnt=1)
    sns.regplot(x="Attendance", y=y, data=df.sample(2000, random_state=SEED), scatter=False, color="red", ax=a)
    r = df.Attendance.corr(df[y]); a.set_title(f"Attendance vs {y}  (r = {r:+.3f})")
save("02_attendance_vs_marks")

bins = pd.cut(df.Attendance, [0, 50, 60, 70, 80, 90, 100])
fig, ax = plt.subplots(figsize=(7, 4))
df.groupby(bins, observed=True)[["Total_Score", "Performance_Index"]].mean().plot(kind="bar", ax=ax)
ax.set_ylim(50, 85); ax.set_title("Mean marks by attendance band"); ax.set_xlabel("Attendance band (%)")
save("03_marks_by_attendance_band")

fig, ax = plt.subplots(figsize=(9, 7))
corr_cols = ["Attendance"] + coursework + ["Participation_Score", "Total_Score", "Performance_Index",
             "math_score", "reading_score", "writing_score", "science_score", "test_preparation_course"]
sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, annot_kws={"size": 7}, ax=ax)
ax.set_title("Correlation matrix")
save("04_correlation_heatmap")

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
sns.boxplot(x="Department", y="Performance_Index", data=df, ax=ax[0]); ax[0].tick_params(axis="x", rotation=20)
ax[0].set_title("Performance index by department")
sns.boxplot(x="test_preparation_course", y="Performance_Index", data=df, ax=ax[1])
ax[1].set_title("Performance index by prep course (0=no, 1=yes)")
save("05_department_prepcourse")

# %% ------------------------------------------------------------------
# 4. RISK DEFINITION (Low / Medium / High)
# ---------------------------------------------------------------------
# Two independent warning signals, each scored 0 / 1 / 2 points:
#   Attendance:        >=75 -> 0 | 60-75 -> 1 | <60 -> 2
#   Performance index: >=70 -> 0 | 60-70 -> 1 | <60 -> 2
# Total points: 0-1 = LOW, 2 = MEDIUM, 3-4 = HIGH
# Rationale: one weak area alone (<=1 point) is a watch item; a student who is
# clearly weak on one signal AND borderline on the other (>=3) needs intervention.
# NOTE: Total_Score/Grade are NOT used for the label because the audit above shows
# they are unrelated to the component scores (data inconsistency).
def pts(value, ok, warn):
    return np.where(value >= ok, 0, np.where(value >= warn, 1, 2))

df["Risk_Points"] = pts(df.Attendance, 75, 60) + pts(df.Performance_Index, 70, 60)
df["Risk"] = pd.cut(df.Risk_Points, [-1, 1, 2, 4], labels=["Low", "Medium", "High"])
print("\n--- Risk distribution (rule-based) ---")
print(df.Risk.value_counts().reindex(["Low", "Medium", "High"]).to_string())

fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x="Risk", data=df, order=["Low", "Medium", "High"], palette=["#4CAF50", "#FFC107", "#F44336"], ax=ax)
ax.set_title("Students per risk level")
save("06_risk_distribution")

# %% ------------------------------------------------------------------
# 5. FEATURE SELECTION + ML PREPARATION
# ---------------------------------------------------------------------
# Design choice: an EARLY-WARNING model. It must work before the final exam and
# projects exist, so those (and Total_Score) are excluded. They also build the label,
# so using them would be data leakage. The model predicts risk from what is known
# mid-semester. We also include demographic/other-subject columns to let the
# model prove whether they matter (feature importance later).
# Step A: start with a FULL candidate set (core + demographic + unrelated subject scores)
full_features = ["Attendance", "Midterm_Score", "Assignments_Avg", "Quizzes_Avg", "Participation_Score",
                 "test_preparation_course", "math_score", "reading_score", "writing_score", "science_score", "Age"]
X_full = df[full_features].join(pd.get_dummies(df[["Gender", "Department"]], drop_first=True).astype(int))

# Step B: the EDA showed the extra columns have ~0 correlation with anything, and they are not
# part of the risk definition. We TEST whether they help (see comparison below) and keep only the CORE set.
core_features = ["Attendance", "Midterm_Score", "Assignments_Avg", "Quizzes_Avg", "Participation_Score"]
X = df[core_features].copy()
y = df["Risk"].astype(str)

# Stratified split keeps class proportions identical in train and test.
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
print(f"\nTrain {X_tr.shape}, Test {X_te.shape}")

# %% ------------------------------------------------------------------
# 6. MODEL TRAINING + COMPARISON (5-fold stratified CV, macro-F1)
# ---------------------------------------------------------------------
# Macro-F1 treats all three classes equally, so a rare 'High' class is not ignored.
models = {
    "Logistic Regression": make_pipeline(StandardScaler(),
                                         LogisticRegression(max_iter=2000, class_weight="balanced")),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=30, class_weight="balanced", random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced",
                                            n_jobs=-1, random_state=SEED),
}
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
cv_scores = {}
print("\n--- Cross-validation (macro-F1) ---")
for name, m in models.items():
    s = cross_val_score(m, X_tr, y_tr, cv=cv, scoring="f1_macro")
    cv_scores[name] = s
    print(f"{name:20s} {s.mean():.3f} +/- {s.std():.3f}")

best_name = max(cv_scores, key=lambda k: cv_scores[k].mean())

# Feature-selection check: same Random Forest, FULL (15 features) vs CORE (5 features)
rf = models["Random Forest"]
f_full = cross_val_score(rf, X_full.loc[X_tr.index], y_tr, cv=cv, scoring="f1_macro")
f_core = cross_val_score(rf, X_tr, y_tr, cv=cv, scoring="f1_macro")
print(f"\nFeature-selection check (Random Forest CV macro-F1): "
      f"full {X_full.shape[1]} features = {f_full.mean():.3f} | core {X_tr.shape[1]} features = {f_core.mean():.3f}")
best = models[best_name].fit(X_tr, y_tr)
print("Selected model:", best_name)

# %% ------------------------------------------------------------------
# 7. EVALUATION ON HELD-OUT TEST SET
# ---------------------------------------------------------------------
labels = ["Low", "Medium", "High"]
pred = best.predict(X_te)
acc = accuracy_score(y_te, pred)
p, r, f, _ = precision_recall_fscore_support(y_te, pred, labels=labels, average="macro")
print(f"\n--- Test results ({best_name}) ---")
print(f"Accuracy {acc:.3f} | macro Precision {p:.3f} | macro Recall {r:.3f} | macro F1 {f:.3f}")
print(classification_report(y_te, pred, labels=labels, digits=3))

cm = confusion_matrix(y_te, pred, labels=labels)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax[0], cmap="Blues", colorbar=False)
ax[0].set_title("Confusion matrix (counts)")
ConfusionMatrixDisplay(confusion_matrix(y_te, pred, labels=labels, normalize="true"), display_labels=labels
                       ).plot(ax=ax[1], cmap="Blues", colorbar=False, values_format=".2f")
ax[1].set_title("Confusion matrix (row-normalised = recall)")
save("07_confusion_matrix")

# Model comparison chart
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.barh(list(cv_scores), [v.mean() for v in cv_scores.values()], xerr=[v.std() for v in cv_scores.values()])
ax.set_xlabel("CV macro-F1"); ax.set_title("Model comparison")
save("08_model_comparison")

# Feature importance (what drives risk?)
if hasattr(best, "feature_importances_"):
    imp = pd.Series(best.feature_importances_, index=X.columns)
else:                                                    # logistic regression: mean |coef|
    imp = pd.Series(np.abs(best[-1].coef_).mean(axis=0), index=X.columns)
imp = imp.sort_values()
fig, ax = plt.subplots(figsize=(7, 6)); imp.plot(kind="barh", ax=ax); ax.set_title("Feature importance")
save("09_feature_importance")
print("Top drivers:", imp.sort_values(ascending=False).head(4).round(3).to_dict())

# %% ------------------------------------------------------------------
# 8. PREDICTION + RECOMMENDATION ENGINE
# ---------------------------------------------------------------------
# Out-of-fold predictions: every student is scored by a model that never saw them,
# so the report is not inflated by in-sample memorisation.
df["Predicted_Risk"] = cross_val_predict(models[best_name], X, y, cv=cv)

AREAS = {"Midterm_Score": "midterm exam preparation", "Assignments_Avg": "assignment completion",
         "Quizzes_Avg": "regular quiz practice", "Participation_Pct": "class participation",
         "Projects_Score": "project work", "Final_Score": "final exam preparation"}

def recommend(row):
    """Rule-based advice from the student's weakest areas (kept simple and explainable)."""
    risk = row["Predicted_Risk"]
    if risk == "Low":
        return "On track. Keep up current attendance and study habits."
    tips = []
    if row.Attendance < 60:   tips.append("Urgently improve attendance (below 60%)")
    elif row.Attendance < 75: tips.append("Improve attendance (below 75%)")
    weak = sorted([k for k in AREAS if row[k] < 65], key=lambda k: row[k])[:2]   # 2 weakest areas under 65
    if weak:                  tips.append("focus on " + " and ".join(AREAS[k] for k in weak))
    if not tips:              tips.append("Monitor progress and maintain current performance")
    text = "; ".join(tips)
    text = text[0].upper() + text[1:] + "."
    if risk == "High":        text += " Schedule a meeting with an academic advisor."
    return text

df["Recommendation"] = df.apply(recommend, axis=1)

def show_student(row):
    """Prints a student in the format requested by the brief."""
    print(f"\nStudent: {row.Student_ID} ({row.First_Name} {row.Last_Name})\n"
          f"Attendance: {row.Attendance:.0f}%\nMarks: {row.Total_Score:.0f} "
          f"(performance index {row.Performance_Index:.0f})\nRisk: {row.Predicted_Risk.upper()}\n"
          f"Recommendation:\n{row.Recommendation}")

print("\n=== Sample predictions ===")
for lvl in ["High", "Medium", "Low"]:
    show_student(df[df.Predicted_Risk == lvl].sample(1, random_state=SEED).iloc[0])

# %% ------------------------------------------------------------------
# 9. AUTOMATED REPORT (bonus)
# ---------------------------------------------------------------------
order = {"High": 0, "Medium": 1, "Low": 2}
report = (df.assign(Student=df.Student_ID + " - " + df.First_Name + " " + df.Last_Name,
                    **{"Attendance (%)": df.Attendance.round(1), "Marks": df.Total_Score.round(1),
                       "Performance Index": df.Performance_Index.round(1)},
                    Risk=df.Predicted_Risk.str.upper())
            [["Student", "Attendance (%)", "Marks", "Performance Index", "Risk", "Recommendation"]]
            .sort_values("Risk", key=lambda s: s.str.title().map(order), kind="stable"))
report.to_csv(f"{OUT}/student_risk_report.csv", index=False)
try:
    with pd.ExcelWriter(f"{OUT}/student_risk_report.xlsx") as xw:
        report.to_excel(xw, sheet_name="All students", index=False)
        report[report.Risk == "HIGH"].to_excel(xw, sheet_name="High risk only", index=False)
        report.Risk.value_counts().rename_axis("Risk").reset_index(name="Students").to_excel(xw, sheet_name="Summary", index=False)
except Exception as e:
    print("Excel export skipped:", e)

joblib.dump({"model": best, "features": list(X.columns)}, f"{OUT}/risk_model.joblib")
print(f"\nReport saved ({len(report)} students). High-risk students: {(report.Risk == 'HIGH').sum()}")
print(report.head(5).to_string(index=False))
