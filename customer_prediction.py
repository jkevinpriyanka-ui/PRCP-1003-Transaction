import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load("final_logistic.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Customer Transaction Prediction (200 Features)")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    expected_features = [f"VAR{i}" for i in range(1, 201)]
    missing = [f for f in expected_features if f not in df.columns]
    
    if missing:
        st.error(f"Missing features: {', '.join(missing)}")
    else:
        X_scaled = scaler.transform(df[expected_features])
        probs = model.predict_proba(X_scaled)[:,1]
        labels = ["High-Confidence Transact" if p>0.7 else "Will Transact" if p>0.5 else "Will Not Transact" for p in probs]
        
        df["Probability"] = probs
        df["Prediction"] = labels
        
        st.dataframe(df)
        
        st.download_button("Download Predictions CSV", df.to_csv(index=False).encode(), "predictions.csv", "text/csv")
        
        counts = df["Prediction"].value_counts()
        fig, ax = plt.subplots()
        ax.bar(counts.index, counts.values, color=["skyblue","orange","green"])
        ax.set_ylabel("Number of Customers")
        ax.set_title("Predicted Transaction Counts")
        st.pyplot(fig)

