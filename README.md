# 🎓 Student Academic Risk — Early-Warning System

A machine-learning pipeline that identifies students who may need academic intervention, classifies them as **Low / Medium / High risk**, and generates a personalised recommendation for each student. Results are presented in an interactive HTML dashboard and an automated report.

**🔗 Live dashboard:** ``

---

## ✨ What it does

| Step | What happens |
|---|---|
| 1. Explore | Profiles the data, tests attendance against marks, draws 9 charts |
| 2. Clean | Fixes duplicates, inconsistent labels, invalid values and missing data |
| 3. Define risk | Scores each student with a transparent points system (Low / Medium / High) |
| 4. Model | Compares Logistic Regression, Decision Tree and Random Forest; keeps the best |
| 5. Evaluate | Accuracy, precision, recall, F1, confusion matrix, feature importance |
| 6. Recommend | Gives each student a tailored action based on their weakest areas |
| 7. Report | Exports a Student \| Attendance \| Risk \| Recommendation report (Excel + CSV) and the dashboard |

---

## 📁 Repository structure

```
├── student_risk_pipeline.py     # the full pipeline (heavily commented)
├── dcs_student_data.csv         # input dataset
├── requirements.txt             # Python libraries needed
├── dashboard.html               # interactive dashboard (generated)
├── student_risk_report.xlsx     # automated report: all students / high risk only / summary
├── student_risk_report.csv      # same report as CSV
├── risk_model.joblib            # saved trained Random Forest (proof of the trained model)
└── figures/                     # 9 charts
```

---

## 🚀 Quick start

**Requirements:** Python 3.9 or newer.

```bash
# 1. Clone the repository
git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
cd YOUR-REPO

# 2. Install the libraries
pip install -r requirements.txt

# 3. Run the pipeline
python student_risk_pipeline.py
```

This prints the analysis in the terminal and regenerates the report, charts, saved model and `dashboard.html` in the same folder. Open `dashboard.html` in any browser (no internet needed).

To use a different CSV with the same columns:

```bash
python student_risk_pipeline.py path/to/your_file.csv
```

---

## 🔍 Key findings

- **Attendance has almost no relationship with marks in this dataset** (Pearson r ≈ −0.01 with `Total_Score`, p = 0.28). The pattern is flat across attendance bands, departments and prep-course status.
- **`Total_Score` and `Grade` look unreliable.** `Total_Score` is uncorrelated with the component scores, and the average `Total_Score` is about 75 for every grade from A to F. They were therefore *not* used to define risk.
- **Data quality problems found and fixed:** 30 duplicate student IDs, inconsistent labels (`CS` vs `Computer Science`, `FEMALE`, ` engineering `), ~200 missing values in each of 8 columns, values outside valid ranges (e.g. attendance 135%), and a stray tab character in `math_score`.

---

## 🧹 Data preprocessing

- Dropped duplicate `Student_ID`s (kept first) → **10,000 students**
- Standardised text categories (case, whitespace, aliases)
- Converted `math_score` to numeric
- Treated out-of-range values as invalid (scores/attendance outside 0–100, participation outside 0–10, age outside 15–40) and imputed them
- Imputed numeric gaps with the **median** (robust to outliers) and categorical gaps with the **mode**
- Removed ID, name and email columns from modelling (no predictive value)
- Stratified 80/20 train/test split so class proportions are preserved

---

## ⚠️ How risk is defined

Each student gets 0, 1 or 2 points on two independent signals:

| Signal | 0 pts | 1 pt | 2 pts |
|---|---|---|---|
| Attendance | ≥ 75% | 60–75% | < 60% |
| Performance index* | ≥ 70 | 60–70 | < 60 |

**Total points → risk level:** 0–1 = 🟢 **Low** · 2 = 🟡 **Medium** · 3–4 = 🔴 **High**

\*The performance index is a weighted average of graded work: midterm 25%, final 30%, assignments 15%, quizzes 10%, projects 15%, participation 5%.

**Rationale:** one weak area alone is a watch item; a student who is clearly weak on one signal *and* borderline on the other needs intervention.

**Result:** 6,220 Low · 2,601 Medium · 1,179 High (predicted risk across all students).

---

## 🤖 Model

**Early-warning design.** The model uses only information available mid-semester: attendance, midterm, assignments, quizzes and participation. Final exam, projects and `Total_Score` are excluded — they are used to build the risk label, so including them would leak the answer, and a model that needs the final exam can't warn anyone early.

**Feature selection.** The same Random Forest was scored with all 15 candidate features and with just these 5 (cross-validated macro-F1):

| Feature set | CV macro-F1 |
|---|---|
| All 15 features | 0.707 |
| Core 5 features | 0.706 |

The extra columns (subject scores, age, gender, department, prep course) add nothing, so they were dropped.

**Model comparison (5-fold stratified CV, macro-F1):**

| Model | CV macro-F1 |
|---|---|
| Logistic Regression | 0.656 |
| Decision Tree | 0.671 |
| **Random Forest** ✅ | **0.706** |

---

## 📊 Results (held-out test set, 2,000 students)

| Metric | Score |
|---|---|
| Accuracy | **80.0%** |
| Macro precision | 0.72 |
| Macro recall | 0.74 |
| Macro F1 | 0.725 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Low | 0.90 | 0.90 | 0.90 | 1,273 |
| Medium | 0.63 | 0.59 | 0.61 | 519 |
| High | 0.62 | 0.73 | 0.67 | 208 |

**What drives risk (feature importance):** attendance (0.52) › midterm (0.20) › assignments (0.11) › participation (0.09) › quizzes (0.08).

High-risk **recall of 0.73** is the number that matters most here: it is the share of truly at-risk students the model catches.

---

## 💡 Example output

```
Student: S1000 (Omar Williams)
Attendance: 52%
Marks: 66 (weighted coursework average)
Risk: HIGH

Recommendation:
Urgently improve attendance (below 60%); focus on class participation and
midterm exam preparation. Schedule a meeting with an academic advisor.
```

Recommendations are rule-based and explainable: they flag low attendance, name the student's two weakest areas below 65, and add an advisor meeting for High-risk students.

The full report (`student_risk_report.xlsx` / `.csv`) lists **Student | Attendance | Marks | Performance Index | Risk | Recommendation**, sorted with High-risk students first. Report risk levels are *out-of-fold* predictions, so every student is scored by a model that never saw them.

---

## 🖥️ Dashboard

`dashboard.html` is a single self-contained file (no server, no internet required) with:

- Summary cards: students analysed, High / Medium risk counts, model accuracy
- Risk distribution donut and risk mix by attendance band
- Attendance-vs-marks chart with the correlation finding
- Model metrics, confusion matrix and feature importance
- **Student lookup** — type an ID (e.g. `S1000`) to see the brief's Student / Attendance / Marks / Risk / Recommendation card
- A searchable, sortable, paginated table of all students with risk and department filters

---

## ⚖️ Limitations

- **The risk label is rule-based.** The dataset has no real "at-risk" outcome, so the model learns the risk definition above. The 80% accuracy shows how well it reproduces that rule early in the semester — it is not evidence of hidden patterns.
- **The data appears synthetic or randomly generated** (near-zero correlations between almost all columns), so the findings should not be generalised to real students.
- **Thresholds (75 / 60 / 70) are judgement calls** and should be tuned with an institution's own policy and outcomes.

---

## 🛠️ Tech stack

Python · pandas · NumPy · scikit-learn · SciPy · matplotlib · seaborn · joblib · openpyxl · HTML/CSS/JavaScript (dashboard)

---

## 📄 License

Add a license of your choice (e.g. MIT) before publishing.
