import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# ------------------------- Page config -------------------------
st.set_page_config(page_title="Healthcare Analytics Dashboard",
                   page_icon="🩺",
                   layout="wide")

st.markdown("<h2 style='color:#007acc;'>Healthcare Analytics</h2>", unsafe_allow_html=True)
st.markdown("---")

# ------------------------- Load Data ---------------------------
@st.cache_data
def load_data():
    usecols = [
        "race", "gender", "age", "time_in_hospital",
        "num_lab_procedures", "num_medications",
        "number_outpatient", "number_emergency", "number_inpatient",
        "readmitted", "patient_nbr"
    ]
    df = pd.read_csv("hospital_readmission.csv", usecols=usecols)

    # Convert age like [10-20) → 15 midpoint
    if "age" in df.columns and df["age"].dtype == "object":
        df["age"] = df["age"].str.extract(r"(\d+)").astype(float)

    # Binary target variable
    df["readmitted_30d"] = df["readmitted"].apply(lambda x: 1 if x == "<30" else 0)

    # Convert categorical columns
    for col in ["race", "gender", "readmitted"]:
        df[col] = df[col].astype("category")

    # Sample large dataset for performance
    if len(df) > 50000:
        df = df.sample(50000, random_state=42)

    return df


df = load_data()

# ------------------------- KPIs -------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", f"{len(df):,}")
col2.metric("Unique Patients", f"{df['patient_nbr'].nunique():,}")
col3.metric("Avg Hospital Stay", f"{df['time_in_hospital'].mean():.2f} days")
col4.metric("Readmission Rate (30d)", f"{df['readmitted_30d'].mean()*100:.2f}%")

st.markdown("---")

# ------------------------- Sidebar Filters -------------------------
with st.sidebar:
    st.header("Filters ⚙️")
    gender_filter = st.multiselect("Gender", options=df["gender"].unique(), default=list(df["gender"].unique()))
    race_filter = st.multiselect("Race", options=df["race"].unique(), default=list(df["race"].unique()))
    age_range = st.slider("Age Range", int(df["age"].min()), int(df["age"].max()), (30, 70))
    show_ml = st.checkbox("Show ML Model Panel", value=True)

df_filtered = df[
    (df["gender"].isin(gender_filter)) &
    (df["race"].isin(race_filter)) &
    (df["age"].between(age_range[0], age_range[1]))
]

# ------------------------- Graphs Section -------------------------
st.subheader("📊 Exploratory Data Analysis")

# First row (4 graphs)
col1, col2, col3, col4 = st.columns(4)

with col1:
    fig1 = px.histogram(df_filtered, x="age", nbins=20, title="Age Distribution", color_discrete_sequence=["#1f77b4"])
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    gender_dist = df_filtered["gender"].value_counts().reset_index()
    gender_dist.columns = ["gender", "count"]
    fig2 = px.bar(gender_dist, x="gender", y="count", title="Gender Distribution", text="count", color="gender")
    st.plotly_chart(fig2, use_container_width=True)

with col3:
    fig3 = px.pie(df_filtered, names="race", title="Race Composition", hole=0.3)
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    admissions_ts = df_filtered.groupby("time_in_hospital").size().reset_index(name="count")
    fig4 = px.line(admissions_ts, x="time_in_hospital", y="count", title="Admissions by Hospital Stay", markers=True)
    st.plotly_chart(fig4, use_container_width=True)

# Second row (4 graphs)
col5, col6, col7, col8 = st.columns(4)

with col5:
    fig5 = px.box(df_filtered, y="num_lab_procedures", color="gender", title="Lab Procedures by Gender")
    st.plotly_chart(fig5, use_container_width=True)

with col6:
    fig6 = px.violin(df_filtered, y="num_medications", color="race", title="Medication Count by Race", box=True, points="all")
    st.plotly_chart(fig6, use_container_width=True)

with col7:
    avg_inpatient = df_filtered.groupby("time_in_hospital")["number_inpatient"].mean().reset_index()
    fig7 = px.bar(avg_inpatient, x="time_in_hospital", y="number_inpatient", title="Avg Inpatient Visits vs Hospital Stay")
    st.plotly_chart(fig7, use_container_width=True)

with col8:
    fig8 = px.density_heatmap(df_filtered, x="num_medications", y="num_lab_procedures",
                              title="Heatmap: Medications vs Lab Procedures", nbinsx=20, nbinsy=20)
    st.plotly_chart(fig8, use_container_width=True)

# ------------------------- Enhanced Auto-Playing Bubble Chart -------------------------
st.subheader("🌈 Moving Bubble Chart — Auto Dynamic Playback")

bubble_df = (
    df_filtered
    .sample(min(10000, len(df_filtered)), random_state=42)
    .groupby(["time_in_hospital", "race", "gender"])
    .agg({
        "num_medications": "mean",
        "num_lab_procedures": "mean",
        "number_inpatient": "mean"
    })
    .reset_index()
)

# Clean NaN values
bubble_df = bubble_df.dropna(subset=["number_inpatient", "num_medications", "num_lab_procedures"])
bubble_df["number_inpatient"] = bubble_df["number_inpatient"].fillna(bubble_df["number_inpatient"].median())

# Create figure
fig_bubble = px.scatter(
    bubble_df,
    x="num_medications",
    y="num_lab_procedures",
    animation_frame="time_in_hospital",
    animation_group="race",
    size="number_inpatient",
    color="race",
    hover_name="gender",
    size_max=50,
    range_x=[0, bubble_df["num_medications"].max() + 5],
    range_y=[0, bubble_df["num_lab_procedures"].max() + 5],
    title="Dynamic Bubble Chart: Medications vs Lab Procedures Over Hospital Stay Duration"
)

# Improve visuals
fig_bubble.update_traces(marker=dict(opacity=0.8, line=dict(width=1, color="DarkSlateGrey")))
fig_bubble.update_layout(
    transition={'duration': 800, 'easing': 'cubic-in-out'},
    showlegend=True,
)

# 🔁 Auto-play animation without user input
fig_bubble.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 800
fig_bubble.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 500

# Loop animation automatically (simulate infinite playback)
fig_bubble.layout.updatemenus[0].buttons[0].args[1]["mode"] = "immediate"
fig_bubble.layout.updatemenus[0].buttons[0].args[1]["fromcurrent"] = True

# Set autoplay and looping
fig_bubble.layout.updatemenus[0].buttons[0].args[1]["repeat"] = True
fig_bubble.layout.updatemenus[0].showactive = False

st.plotly_chart(fig_bubble, use_container_width=True)

# ------------------------- Machine Learning Section -------------------------
if show_ml:
    st.markdown("---")
    st.subheader("🤖 ML: Predicting 30-Day Readmission")

    df_ml = df_filtered.copy()
    features = [
        "age", "time_in_hospital", "num_lab_procedures",
        "num_medications", "number_outpatient", "number_emergency", "number_inpatient"
    ]
    df_ml = df_ml[features + ["readmitted_30d", "gender", "race"]].dropna()

    # Encoding categorical vars
    le_gender = LabelEncoder()
    df_ml["gender_enc"] = le_gender.fit_transform(df_ml["gender"])
    le_race = LabelEncoder()
    df_ml["race_enc"] = le_race.fit_transform(df_ml["race"])

    X = df_ml[features + ["gender_enc", "race_enc"]]
    y = df_ml["readmitted_30d"]

    test_size = st.slider("Test size (%)", 10, 40, 25)
    model_choice = st.selectbox("Choose ML Model", ["Random Forest", "Logistic Regression", "Decision Tree", "KNN"])

    if st.button("Train Selected Model"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size / 100, random_state=42, stratify=y)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if model_choice == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]
        elif model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=200)
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            probs = model.predict_proba(X_test_scaled)[:, 1]
        elif model_choice == "Decision Tree":
            model = DecisionTreeClassifier(random_state=42, max_depth=6)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]
        else:
            model = KNeighborsClassifier(n_neighbors=5)
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            probs = model.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        st.success(f"✅ {model_choice} Accuracy: {acc:.3f} | ROC AUC: {auc:.3f}")

        prob_df = pd.DataFrame({"True Label": y_test, "Predicted Probability": probs})
        fig_prob = px.histogram(
            prob_df, x="Predicted Probability", color=prob_df["True Label"].astype(str),
            nbins=20, title=f"{model_choice} — Prediction Probability Distribution"
        )
        st.plotly_chart(fig_prob, use_container_width=True)

# ------------------------- Footer -------------------------
st.markdown("---")
st.caption("DSAA Project — Madhura Chavan")
