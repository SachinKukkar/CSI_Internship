import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Optional: Load scaler if you used one
try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except:
    scaler = None

# Page title
st.title("🌿 ML Model Deployment with Streamlit")
st.write("Enter the input features below to get predictions.")

# Input widgets
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.35)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.3)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
input_df = pd.DataFrame(input_data, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])

# Scale input if needed
if scaler:
    input_data = scaler.transform(input_data)

# Prediction
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)

# Output
st.subheader("📊 Model Prediction")
st.write(f"**Predicted Class:** {prediction[0]}")
st.write("**Prediction Probability:**")
st.dataframe(pd.DataFrame(prediction_proba, columns=model.classes_))

# Visualization
st.subheader("🔍 Feature Input Overview")
st.dataframe(input_df)

# Example visualization
st.subheader("📈 Probability Bar Chart")
fig, ax = plt.subplots()
sns.barplot(x=model.classes_, y=prediction_proba[0], ax=ax)
ax.set_ylabel("Probability")
st.pyplot(fig)
