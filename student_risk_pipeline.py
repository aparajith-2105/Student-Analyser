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

# %% ------------------------------------------------------------------
# 10. VISUAL DASHBOARD (one self-contained HTML file - open it in any browser)
# ---------------------------------------------------------------------
import json, base64

def build_dashboard():
    # ---- numbers the dashboard displays ---------------------------------
    pr = precision_recall_fscore_support(y_te, pred, labels=labels)
    class_metrics = [{"cls": l, "p": float(pr[0][i]), "r": float(pr[1][i]), "f": float(pr[2][i]), "n": int(pr[3][i])}
                     for i, l in enumerate(labels)]
    band_labels = ["<50", "50-60", "60-70", "70-80", "80-90", "90-100"]
    band = pd.cut(df.Attendance, [0, 50, 60, 70, 80, 90, 100], labels=band_labels)
    mix = (pd.crosstab(band, df.Predicted_Risk).reindex(columns=labels, fill_value=0)
             .reindex(band_labels).fillna(0).astype(int))
    rec_list = sorted(df.Recommendation.unique())
    rec_idx = {r: i for i, r in enumerate(rec_list)}
    meta = df.loc[report.index]
    rows = [[sid.split(" - ")[0], sid.split(" - ")[1], dp, float(a), float(pi), float(m), rk.title(), rec_idx[rc]]
            for sid, dp, a, m, pi, rk, rc in zip(report.Student, meta.Department, report["Attendance (%)"],
                                                  report.Marks, report["Performance Index"], report.Risk,
                                                  report.Recommendation)]
    fig_b64 = base64.b64encode(open(f"{FIG}/02_attendance_vs_marks.png", "rb").read()).decode()
    data = {
        "total": int(len(df)),
        "risk": {l: int((df.Predicted_Risk == l).sum()) for l in labels},
        "acc": float(acc), "macro_f1": float(f), "classes": class_metrics,
        "cm": cm.tolist(), "labels": labels,
        "imp": [[k, float(v)] for k, v in imp.sort_values(ascending=False).items()],
        "bands": band_labels, "mix": {l: mix[l].tolist() for l in labels},
        "r_total": float(df.Attendance.corr(df.Total_Score)), "r_index": float(df.Attendance.corr(df.Performance_Index)),
        "dups": int(raw.Student_ID.duplicated().sum()), "model": best_name,
        "cv": {k: float(v.mean()) for k, v in cv_scores.items()},
        "f_full": float(f_full.mean()), "f_core": float(f_core.mean()),
        "recs": rec_list, "rows": rows, "scatter": fig_b64,
    }
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = DASH_TEMPLATE.replace("/*__DATA__*/", "const D = " + payload + ";")
    with open(f"{OUT}/dashboard.html", "w", encoding="utf-8") as fh:
        fh.write(html)
    print("Dashboard saved -> dashboard.html")

DASH_TEMPLATE = r'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Student Risk Early-Warning Dashboard</title>
<style>
:root{--bg:#f4f6fb;--card:#fff;--ink:#1e2433;--mute:#6b7488;--line:#e6e9f2;--brand:#4f46e5;
--low:#16a34a;--med:#f59e0b;--high:#ef4444;--lowbg:#dcfce7;--medbg:#fef3c7;--highbg:#fee2e2}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font-family:Inter,"Segoe UI",system-ui,-apple-system,Roboto,sans-serif;line-height:1.5}
header{background:linear-gradient(135deg,#312e81,#4f46e5 55%,#7c3aed);color:#fff;padding:42px 24px 90px}
.wrap{max-width:1180px;margin:0 auto;padding:0 20px}
header h1{margin:0 0 6px;font-size:30px;letter-spacing:-.5px}header p{margin:0;opacity:.85}
.pill{display:inline-block;background:rgba(255,255,255,.18);padding:4px 12px;border-radius:99px;font-size:12px;margin-bottom:12px}
main{margin-top:-62px;padding-bottom:50px}
.grid{display:grid;gap:18px}.g4{grid-template-columns:repeat(4,1fr)}.g2{grid-template-columns:1fr 1fr}.g3{grid-template-columns:minmax(0,1fr) minmax(0,1.25fr) minmax(0,1fr)}
@media(max-width:900px){.g4{grid-template-columns:1fr 1fr}.g2,.g3{grid-template-columns:1fr}}
.card{background:var(--card);border-radius:16px;padding:22px;box-shadow:0 1px 3px rgba(20,30,70,.06),0 8px 24px rgba(20,30,70,.05)}
.card h3{margin:0 0 4px;font-size:16px}.sub{color:var(--mute);font-size:13px;margin:0 0 16px}
section{margin-top:18px}
.kpi .lab{color:var(--mute);font-size:13px;font-weight:600;text-transform:uppercase;letter-spacing:.5px}
.kpi .val{font-size:34px;font-weight:800;margin-top:6px;letter-spacing:-1px}
.kpi .note{font-size:12.5px;color:var(--mute)}
.k-high{border-top:4px solid var(--high)}.k-med{border-top:4px solid var(--med)}.k-low{border-top:4px solid var(--low)}.k-brand{border-top:4px solid var(--brand)}
.donutwrap{display:flex;align-items:center;gap:26px;flex-wrap:wrap}
.donut{width:170px;height:170px;border-radius:50%;position:relative;flex:none}
.donut::after{content:"";position:absolute;inset:30px;background:#fff;border-radius:50%}
.donut b{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;z-index:2;font-size:24px}
.donut b small{font-size:11px;color:var(--mute);font-weight:500}
.legend div{display:flex;align-items:center;gap:9px;margin:7px 0;font-size:14px}
.dot{width:11px;height:11px;border-radius:50%}
.bars .row{display:flex;align-items:center;gap:10px;margin:9px 0;font-size:13px}
.bars .nm{width:62px;color:var(--mute);text-align:right}
.stack{flex:1;height:22px;border-radius:7px;overflow:hidden;display:flex;background:#eef0f6}
.stack i{display:block;height:100%}
.callout{background:#eef2ff;border-left:4px solid var(--brand);border-radius:10px;padding:14px 16px;font-size:14px;margin-top:12px}
.warn{background:#fff7ed;border-left-color:var(--med)}
img.sc{width:100%;border-radius:10px;border:1px solid var(--line)}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th{background:#f7f8fc;text-align:left;padding:10px 12px;color:var(--mute);font-size:12px;text-transform:uppercase;letter-spacing:.4px;cursor:pointer;white-space:nowrap}#tbl th{position:sticky;top:0}
td{padding:10px 12px;border-top:1px solid var(--line);vertical-align:top}
tr.clk:hover td{background:#f7f8ff;cursor:pointer}
.tag{display:inline-block;padding:2px 11px;border-radius:99px;font-size:12px;font-weight:700}
.t-Low{background:var(--lowbg);color:#166534}.t-Medium{background:var(--medbg);color:#92400e}.t-High{background:var(--highbg);color:#991b1b}
#cls th,#cls td{padding:10px 5px;font-size:13px}#cls th{font-size:11px}.cm td,.cm th{text-align:center;padding:10px 4px;font-size:14px;cursor:default}.cm th{font-size:11px}
.cm td{font-weight:700;border:2px solid #fff;border-radius:6px}
.imp .row{display:flex;align-items:center;gap:10px;margin:8px 0;font-size:13px}
.imp .nm{width:92px;color:var(--mute);flex:none}.imp .track{flex:1;min-width:50px;background:#eef0f6;border-radius:7px;height:14px}.imp .bar{height:14px;border-radius:7px;background:linear-gradient(90deg,#4f46e5,#8b5cf6)}.imp .row span:last-child{width:34px;text-align:right}
.tools{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px}
input,select,button{font:inherit;padding:9px 13px;border:1px solid var(--line);border-radius:10px;background:#fff}
input{min-width:230px}button{cursor:pointer;background:var(--brand);color:#fff;border:0;font-weight:600}
button.ghost{background:#eef0f6;color:var(--ink)}button:disabled{opacity:.4;cursor:default}
.scroll{max-height:560px;overflow:auto;border:1px solid var(--line);border-radius:12px}
.pager{display:flex;align-items:center;justify-content:space-between;margin-top:12px;font-size:13px;color:var(--mute)}
.student{border-radius:14px;padding:22px;color:#1e2433;margin-top:6px}
.student.Low{background:var(--lowbg)}.student.Medium{background:var(--medbg)}.student.High{background:var(--highbg)}
.student .line{font-size:16px;margin:3px 0}.student .big{font-size:22px;font-weight:800}
.meter{height:10px;background:rgba(0,0,0,.1);border-radius:6px;overflow:hidden;margin:4px 0 12px}.meter i{display:block;height:100%;background:#4f46e5}
.two{display:grid;grid-template-columns:1fr 1fr;gap:12px}
ul.n{margin:6px 0 0 18px;padding:0;font-size:14px}ul.n li{margin:5px 0}
footer{text-align:center;color:var(--mute);font-size:12.5px;padding:10px 0 30px}
</style></head><body>
<header><div class="wrap"><span class="pill">Early-Warning System</span>
<h1>Student Academic Risk Dashboard</h1>
<p>Who needs academic intervention, why, and what to do about it.</p></div></header>
<main class="wrap">
<div class="grid g4" id="kpis"></div>

<section class="grid g2">
 <div class="card"><h3>Risk distribution</h3><p class="sub">Predicted risk level across all students</p>
  <div class="donutwrap"><div class="donut" id="donut"><b id="donutN"></b></div><div class="legend" id="legend"></div></div></div>
 <div class="card"><h3>Risk mix by attendance</h3><p class="sub">Share of students at each risk level, per attendance band (%)</p>
  <div class="bars" id="bands"></div></div>
</section>

<section class="grid g2">
 <div class="card"><h3>Attendance vs marks</h3><p class="sub">Each cell groups many students; the red line is the trend</p>
  <img class="sc" id="scatter" alt="attendance vs marks"><div class="callout warn" id="corrNote"></div></div>
 <div class="card"><h3>How risk is defined</h3><p class="sub">Two signals, 0&ndash;2 points each</p>
  <table><tr><th>Signal</th><th>0 pts</th><th>1 pt</th><th>2 pts</th></tr>
  <tr><td>Attendance</td><td>&ge; 75%</td><td>60&ndash;75%</td><td>&lt; 60%</td></tr>
  <tr><td>Performance index*</td><td>&ge; 70</td><td>60&ndash;70</td><td>&lt; 60</td></tr></table>
  <ul class="n"><li><span class="tag t-Low">Low</span> 0&ndash;1 points</li><li><span class="tag t-Medium">Medium</span> 2 points</li><li><span class="tag t-High">High</span> 3&ndash;4 points</li></ul>
  <div class="callout">*Weighted average of midterm 25%, final 30%, assignments 15%, quizzes 10%, projects 15%, participation 5%.
  The dataset's own <b>Total_Score</b> and <b>Grade</b> columns don't follow the component scores, so they are not used to define risk.</div></div>
</section>

<section class="card"><h3>Model performance</h3><p class="sub" id="modelSub"></p>
 <div class="grid g3">
  <div><b>Per-class metrics</b><table style="margin-top:8px" id="cls"></table></div>
  <div><b>Confusion matrix</b> <span class="sub">(rows = actual, colour = recall)</span><table class="cm" id="cm" style="margin-top:8px"></table></div>
  <div class="imp"><b>What drives risk</b><div id="imp" style="margin-top:8px"></div></div>
 </div>
 <div class="callout" id="fsNote"></div></section>

<section class="card"><h3>Look up a student</h3><p class="sub">Type a Student ID (e.g. S1000) or click any row in the table below</p>
 <div class="tools"><input id="lookup" placeholder="Student ID, e.g. S1000"><button onclick="doLookup()">Show</button></div>
 <div id="studentCard"></div></section>

<section class="card"><h3>Student report</h3><p class="sub">Student | Attendance | Risk | Recommendation &mdash; searchable and sortable</p>
 <div class="tools"><input id="q" placeholder="Search name, ID, department...">
  <select id="rf"><option value="">All risk levels</option><option>High</option><option>Medium</option><option>Low</option></select>
  <select id="df"><option value="">All departments</option></select></div>
 <div class="scroll"><table id="tbl"><thead><tr>
  <th onclick="sortBy(0)">Student</th><th onclick="sortBy(2)">Dept</th><th onclick="sortBy(3)">Attendance</th>
  <th onclick="sortBy(4)">Marks</th><th onclick="sortBy(6)">Risk</th><th>Recommendation</th></tr></thead><tbody id="tb"></tbody></table></div>
 <div class="pager"><span id="info"></span><span><button class="ghost" id="prev">&larr; Prev</button> <button class="ghost" id="next">Next &rarr;</button></span></div>
</section>
</main>
<footer>Generated by student_risk_pipeline.py &middot; Risk labels are rule-based (no real &ldquo;at-risk&rdquo; outcome exists in the data)</footer>

<script>
/*__DATA__*/
const C={Low:"#16a34a",Medium:"#f59e0b",High:"#ef4444"}, $=id=>document.getElementById(id), pct=x=>(x*100).toFixed(1)+"%";
// KPIs
const R=D.risk;
$("kpis").innerHTML=[
 ["k-brand","Students analysed",D.total.toLocaleString(),"after cleaning duplicates & errors"],
 ["k-high","High risk",R.High.toLocaleString(),pct(R.High/D.total)+" of students"],
 ["k-med","Medium risk",R.Medium.toLocaleString(),pct(R.Medium/D.total)+" of students"],
 ["k-low","Model accuracy",pct(D.acc),D.model+" · macro-F1 "+D.macro_f1.toFixed(2)]
].map(k=>`<div class="card kpi ${k[0]}"><div class="lab">${k[1]}</div><div class="val">${k[2]}</div><div class="note">${k[3]}</div></div>`).join("");
// Donut
let a=0,seg=[];D.labels.slice().reverse().forEach(l=>{const p=R[l]/D.total*100;seg.push(`${C[l]} ${a}% ${a+p}%`);a+=p});
$("donut").style.background=`conic-gradient(${seg.join(",")})`;$("donutN").innerHTML=`${D.total.toLocaleString()}<small>students</small>`;
$("legend").innerHTML=["High","Medium","Low"].map(l=>`<div><span class="dot" style="background:${C[l]}"></span><b>${l}</b>&nbsp;${R[l].toLocaleString()} (${pct(R[l]/D.total)})</div>`).join("");
// Bands
$("bands").innerHTML=D.bands.map((b,i)=>{const t=D.labels.reduce((s,l)=>s+D.mix[l][i],0)||1;if(t==1&&D.labels.every(l=>D.mix[l][i]==0))return "";
 return `<div class="row"><span class="nm">${b}%</span><div class="stack">${["Low","Medium","High"].map(l=>`<i style="width:${D.mix[l][i]/t*100}%;background:${C[l]}" title="${l}: ${D.mix[l][i]}"></i>`).join("")}</div><span style="width:44px">${(D.mix.High[i]/t*100).toFixed(0)}% H</span></div>`}).join("")+
 `<div class="sub" style="margin-top:12px">Green = Low · Amber = Medium · Red = High. Right-hand number is the % High risk. Lower attendance means a much higher chance of High risk, because attendance is part of the risk definition.</div>`;
// Scatter + note
$("scatter").src="data:image/png;base64,"+D.scatter;
$("corrNote").innerHTML=`<b>Finding:</b> attendance has almost no relationship with marks in this dataset (r = ${D.r_total.toFixed(3)} with Total_Score, ${D.r_index.toFixed(3)} with the performance index). Attendance still drives <i>risk</i> because it is scored separately.`;
// Model
$("modelSub").textContent=`${D.model} trained on 5 early-semester features (attendance, midterm, assignments, quizzes, participation); evaluated on a held-out 20% test set.`;
$("cls").innerHTML=`<tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>n</th></tr>`+D.classes.map(c=>`<tr><td><span class="tag t-${c.cls}">${c.cls}</span></td><td>${c.p.toFixed(2)}</td><td>${c.r.toFixed(2)}</td><td>${c.f.toFixed(2)}</td><td>${c.n}</td></tr>`).join("");
$("cm").innerHTML=`<tr><th></th>${D.labels.map(l=>`<th>Pred<br>${l}</th>`).join("")}</tr>`+D.cm.map((r,i)=>{const s=r.reduce((x,y)=>x+y,0)||1;
 return `<tr><th>Actual<br>${D.labels[i]}</th>${r.map((v,j)=>{const k=v/s;return `<td style="background:rgba(79,70,229,${(0.08+k*0.85).toFixed(2)});color:${k>.45?"#fff":"#1e2433"}">${v}</td>`}).join("")}</tr>`}).join("");
const mx=Math.max(...D.imp.map(x=>x[1]));
$("imp").innerHTML=D.imp.map(([k,v])=>`<div class="row"><span class="nm">${k.replace("_Score","").replace("_Avg","").replace("_"," ")}</span><div class="track"><div class="bar" style="width:${v/mx*100}%"></div></div><span>${v.toFixed(2)}</span></div>`).join("");
$("fsNote").innerHTML=`<b>Feature selection:</b> the same model scored ${D.f_full.toFixed(3)} CV macro-F1 with all 15 candidate features and ${D.f_core.toFixed(3)} with just these 5, so the unrelated columns (subject scores, age, gender, department, prep course) were dropped. <b>Caveat:</b> the model learns a rule-based risk definition, so accuracy shows how well it reproduces that rule early in the semester.`;
// Table
const depts=[...new Set(D.rows.map(r=>r[2]))].sort();$("df").innerHTML+=depts.map(d=>`<option>${d}</option>`).join("");
let rows=D.rows,page=0,PS=25,sortCol=null,asc=true;const ord={High:0,Medium:1,Low:2};
function view(){const q=$("q").value.toLowerCase(),rf=$("rf").value,df=$("df").value;
 rows=D.rows.filter(r=>(!rf||r[6]==rf)&&(!df||r[2]==df)&&(!q||(r[0]+" "+r[1]+" "+r[2]).toLowerCase().includes(q)));
 if(sortCol!==null)rows=rows.slice().sort((x,y)=>{let a=x[sortCol],b=y[sortCol];if(sortCol==6){a=ord[a];b=ord[b]}return(a>b?1:a<b?-1:0)*(asc?1:-1)});
 const pages=Math.max(1,Math.ceil(rows.length/PS));page=Math.min(page,pages-1);
 $("tb").innerHTML=rows.slice(page*PS,page*PS+PS).map(r=>`<tr class="clk" onclick="show('${r[0]}',1)"><td><b>${r[1]}</b><br><span style="color:var(--mute)">${r[0]}</span></td><td>${r[2]}</td><td>${r[3].toFixed(1)}%</td><td>${r[4].toFixed(1)}</td><td><span class="tag t-${r[6]}">${r[6]}</span></td><td>${D.recs[r[7]]}</td></tr>`).join("");
 $("info").textContent=`${rows.length.toLocaleString()} students · page ${page+1} of ${pages}`;$("prev").disabled=page==0;$("next").disabled=page>=pages-1}
function sortBy(c){asc=sortCol===c?!asc:true;sortCol=c;view()}
["q","rf","df"].forEach(i=>$(i).addEventListener("input",()=>{page=0;view()}));
$("prev").onclick=()=>{page--;view()};$("next").onclick=()=>{page++;view()};
// Student card (format requested in the brief)
function show(id,sc){const r=D.rows.find(x=>x[0].toLowerCase()==id.trim().toLowerCase());
 if(!r){$("studentCard").innerHTML=`<div class="callout warn">No student found with ID "${id}".</div>`;return}
 $("lookup").value=r[0];
 $("studentCard").innerHTML=`<div class="student ${r[6]}"><div class="two"><div>
 <div class="line"><b>Student:</b> ${r[0]} &mdash; ${r[1]} <span style="opacity:.7">(${r[2]})</span></div>
 <div class="line"><b>Attendance:</b> ${r[3].toFixed(0)}%</div><div class="meter"><i style="width:${r[3]}%"></i></div>
 <div class="line"><b>Marks:</b> ${r[4].toFixed(0)} <span style="opacity:.7">(weighted coursework average; dataset Total_Score: ${r[5].toFixed(0)})</span></div><div class="meter"><i style="width:${r[4]}%"></i></div>
 <div class="line big">Risk: ${r[6].toUpperCase()}</div></div>
 <div><div class="line"><b>Recommendation:</b></div><div class="line">${D.recs[r[7]]}</div></div></div></div>`;
 if(sc)$("studentCard").scrollIntoView({behavior:"smooth",block:"center"})}
function doLookup(){show($("lookup").value,1)}$("lookup").addEventListener("keydown",e=>{if(e.key=="Enter")doLookup()});
view();show(D.rows[0][0],0);
</script></body></html>'''

build_dashboard()
