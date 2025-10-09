# Streamlit-Healthcare-Analysis
🩺 Healthcare Analytics Dashboard

An interactive Streamlit dashboard for exploring hospital readmission data and predicting 30-day readmissions using machine learning.

🚀 Overview

This project analyzes healthcare data with visual insights and predictive modeling.
It helps understand patient demographics, hospital stay patterns, and readmission risks.

📊 Features

Interactive KPIs (Total Records, Avg Stay, Readmission Rate)

8 EDA graphs (age, gender, race, etc.)

Animated Moving Bubble Chart

Built-in ML models (Random Forest, Logistic Regression, Decision Tree, KNN)

Accuracy and ROC AUC metrics

⚙️ Setup

Clone the repo:

git clone https://github.com/<your-username>/healthcare-dashboard.git
cd healthcare-dashboard


Install dependencies:

pip install -r requirements.txt


Run the app:

streamlit run app.py

🧠 Dataset

File: hospital_readmission.csv
Key columns: age, gender, race, time_in_hospital, num_medications, num_lab_procedures, readmitted
Target: readmitted_30d (1 = readmitted within 30 days)

🤖 ML Models

Random Forest

Logistic Regression

Decision Tree

KNN

Each model displays accuracy and ROC AUC scores.

