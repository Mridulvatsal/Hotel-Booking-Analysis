
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Streamlit page setup
st.set_page_config(layout="wide")
st.title("🏨 Hotel Booking Analysis & Cancellation Prediction Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/DELL/Downloads/final-222/Hotel Bookings.csv")
    # Fix missing values safely
    if 'agent' in df.columns:
        df['agent'] = df['agent'].fillna(0)
    if 'company' in df.columns:
        df['company'] = df['company'].fillna(0)
    df.drop_duplicates(inplace=True)
    if 'lead_time' in df.columns:
        df = df[df['lead_time'] < 500]
    return df

# Load data
df = load_data()
st.success(f"✅ Loaded data with shape {df.shape}")

# 1️⃣ Bookings by Month
if 'arrival_date_month' in df.columns:
    st.subheader("📆 Monthly Bookings")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.countplot(x='arrival_date_month', data=df, order=sorted(df['arrival_date_month'].unique()), ax=ax)
    ax.set_title("Bookings by Month")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

# 2️⃣ Cancellation Rate
if 'is_canceled' in df.columns:
    st.subheader("🚫 Cancellation Rate")
    fig, ax = plt.subplots()
    sns.countplot(x='is_canceled', data=df, ax=ax)
    ax.set_title("Cancellations (0 = No, 1 = Yes)")
    st.pyplot(fig)

# 3️⃣ Lead Time vs Cancellation
if {'lead_time', 'is_canceled'}.issubset(df.columns):
    st.subheader("🕒 Lead Time vs Cancellation")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(x='is_canceled', y='lead_time', data=df, ax=ax)
    ax.set_title("Lead Time by Cancellation")
    ax.set_xticklabels(["Not Canceled", "Canceled"])
    st.pyplot(fig)

# 4️⃣ Heatmap
st.subheader("📊 Correlation Heatmap")
fig, ax = plt.subplots(figsize=(14, 10))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
st.pyplot(fig)

# 5️⃣ ML Model Training
model_features = ['lead_time', 'booking_changes', 'previous_cancellations', 'adr', 'total_of_special_requests']
if all(col in df.columns for col in model_features + ['is_canceled']):
    st.header("🤖 Cancellation Prediction Model")
    X = df[model_features]
    y = df['is_canceled']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    st.subheader("📋 Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.subheader("📊 Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("📈 ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba):.2f}")
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)

    # Revenue Risk
    st.header("💰 High-Risk Revenue Loss Estimation")
    df['cancel_prob'] = model.predict_proba(X)[:, 1]
    df['expected_loss'] = df['adr'] * df['cancel_prob']
    high_risk = df[(df['cancel_prob'] > 0.8) & (df['adr'] > df['adr'].median())]

    st.subheader("⚠️ High-Risk Bookings")
    st.write(high_risk[['hotel', 'adr', 'cancel_prob', 'expected_loss']].head())

    if 'arrival_date_month' in df.columns:
        monthly_loss = high_risk.groupby('arrival_date_month')['expected_loss'].sum().sort_index()
        st.subheader("📅 Monthly Revenue Loss")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(x=monthly_loss.index, y=monthly_loss.values, ax=ax)
        ax.set_title("Estimated Monthly Revenue Loss")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)
else:
    st.error("❌ Missing required columns for training model.")
