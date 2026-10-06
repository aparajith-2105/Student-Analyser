"""
Demo: load the saved model (risk_model.joblib) and predict a student's risk.

Run:  python predict_student.py
Then type the 5 values when asked (press Enter to use the example value shown).
"""
import os
import joblib
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# 1) Load the trained model that the main pipeline saved.
saved = joblib.load(os.path.join(HERE, "risk_model.joblib"))
model, features = saved["model"], saved["features"]
print("Loaded:", type(model).__name__, "| inputs:", features)

# 2) Ask for the student's details (Enter = use the example value).
examples = {"Attendance": 55, "Midterm_Score": 58, "Assignments_Avg": 62,
            "Quizzes_Avg": 60, "Participation_Score": 5}
student = {}
for name in features:
    text = input(f"{name} [{examples[name]}]: ").strip()
    student[name] = float(text) if text else examples[name]

# 3) Predict the risk level.
risk = model.predict(pd.DataFrame([student])[features])[0]

# 4) Simple, explainable recommendation (same idea as the main pipeline).
if risk == "Low":
    advice = "On track. Keep up current attendance and study habits."
else:
    tips = []
    if student["Attendance"] < 60:
        tips.append("Urgently improve attendance (below 60%)")
    elif student["Attendance"] < 75:
        tips.append("Improve attendance (below 75%)")
    weak = {"Midterm_Score": "midterm exam preparation", "Assignments_Avg": "assignment completion",
            "Quizzes_Avg": "regular quiz practice"}
    low = [weak[k] for k in weak if student[k] < 65]
    if student["Participation_Score"] * 10 < 65:
        low.append("class participation")
    if low:
        tips.append("focus on " + " and ".join(low[:2]))
    advice = "; ".join(tips) + "." if tips else "Monitor progress."
    if risk == "High":
        advice += " Schedule a meeting with an academic advisor."

print(f"\nAttendance: {student['Attendance']:.0f}%")
print(f"Risk: {risk.upper()}")
print(f"Recommendation:\n{advice}")
