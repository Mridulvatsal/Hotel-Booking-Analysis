
# ----------------------------------
# 📦 PHASE 1: DATA UNDERSTANDING & CLEANING
# ----------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import streamlit as st

# Load the data
df = pd.read_csv('Hotel Bookings (1) (1) (1).csv')
pd.set_option('display.max_columns', None)
print("Data loaded successfully!")
print(df.shape)
print(df.head())

# Basic info and summary
print(df.info())
print(df.describe(include='all'))

# Handle missing values
df.dropna(subset=['children', 'country'], inplace=True)
df['agent'] = df['agent'].fillna(0)
df['company'] = df['company'].fillna(0)
df['children'] = df['children'].fillna(0)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Handle outliers
df = df[df['lead_time'] < 500]

# Save cleaned data
df.to_csv('cleaned_hotel_bookings.csv', index=False)
print("Cleaned data saved.")

# ----------------------------------
# 🔍 PHASE 2: EXPLORATORY DATA ANALYSIS (EDA)
# ----------------------------------

# Create date column
df['month_num'] = df['arrival_date_month'].apply(lambda x: list(calendar.month_name).index(x))
df['arrival_date'] = pd.to_datetime(dict(year=df['arrival_date_year'],
                                         month=df['month_num'],
                                         day=df['arrival_date_day_of_month']))

# Booking trends by month
monthly_bookings = df['arrival_date_month'].value_counts().reindex(
    list(calendar.month_name[1:]), fill_value=0)
monthly_bookings.plot(kind='bar', title='Bookings by Month')
plt.ylabel('Number of Bookings')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Cancellation rate
cancellation_rate = df['is_canceled'].value_counts(normalize=True)
print("Cancellation Rate:\n", cancellation_rate)

# Impact of lead_time on cancellation
sns.boxplot(x='is_canceled', y='lead_time', data=df)
plt.title('Lead Time vs Cancellation')
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ----------------------------------
# 🤖 PHASE 3: CANCELLATION PREDICTION MODEL
# ----------------------------------

# Encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Features and labels
X = df_encoded.drop('is_canceled', axis=1)
y = df_encoded['is_canceled']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Metrics
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))

# ----------------------------------
# 💰 PHASE 4: REVENUE LOSS ESTIMATION
# ----------------------------------

# Assume revenue = adr * stays
df['total_revenue'] = df['adr'] * (df['stays_in_weekend_nights'] + df['stays_in_week_nights'])

# Predict on full data
df_encoded['predicted_cancel'] = model.predict(X)

# Filter high-value at-risk bookings
df_encoded['revenue'] = df['total_revenue']
risky = df_encoded[(df_encoded['predicted_cancel'] == 1) & (df_encoded['revenue'] > df_encoded['revenue'].quantile(0.75))]

# Aggregated loss
monthly_loss = risky.groupby(df['arrival_date_month'])['revenue'].sum()
print("Estimated Revenue Loss by Month:\n", monthly_loss)

# ----------------------------------
# 📊 PHASE 5: DASHBOARD + FINAL REPORT
# ----------------------------------

# Simple Streamlit App (run with: streamlit run <filename.py>)
# st.title("Hotel Booking Analysis Dashboard")
# st.write("This app shows insights and predictions about hotel bookings.")

# st.subheader("Booking Trends")
# st.bar_chart(monthly_bookings)

# st.subheader("Cancellation Rate")
# st.write(cancellation_rate)

# st.subheader("At-Risk High-Value Bookings")
# st.write(risky[['revenue']].describe())

# st.subheader("Total Estimated Revenue Loss")
# st.line_chart(monthly_loss)
