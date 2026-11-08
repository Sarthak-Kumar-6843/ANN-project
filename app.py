import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import json

# -----------------------------
# Load Saved Models
# -----------------------------

model = tf.keras.models.load_model("iris_model.keras")

scaler = joblib.load("scaler.pkl")

encoder = joblib.load("encoder.pkl")

with open("features.json", "r") as f:
    features = json.load(f)


# -----------------------------
# Streamlit Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Iris Flower Prediction",
    page_icon="🌸",
    layout="centered"
)


# -----------------------------
# Application UI
# -----------------------------

st.title("🌸 Iris Flower Classification")

st.write(
    "An Artificial Neural Network (ANN) based "
    "Deep Learning application to predict Iris flower species."
)

st.divider()

st.subheader("Enter Flower Measurements")

# Default values for Iris dataset features
defaults = {
    "SepalLengthCm": 5.1,
    "SepalWidthCm": 3.5,
    "PetalLengthCm": 1.4,
    "PetalWidthCm": 0.2
}

with st.form("prediction_form"):

    input_data = []

    col1, col2 = st.columns(2)

    for i, feature in enumerate(features):

        with col1 if i % 2 == 0 else col2:

            value = st.number_input(
                feature,
                min_value=0.0,
                max_value=10.0,
                value=defaults.get(feature, 1.0),
                step=0.1
            )

            input_data.append(value)

    submit = st.form_submit_button(
        "Predict Iris Species",
        use_container_width=True
    )


# -----------------------------
# Prediction
# -----------------------------

if submit:

    input_array = np.array([input_data])

    # Scale input
    scaled_input = scaler.transform(input_array)

    # ANN Prediction
    prediction = model.predict(scaled_input, verbose=0)

    predicted_class = np.argmax(prediction, axis=1)

    species = encoder.inverse_transform(predicted_class)[0]

    confidence = np.max(prediction) * 100

    # Display Results
    st.divider()

    st.success(f"Predicted Species: {species}")

    st.metric(
        label="Prediction Confidence",
        value=f"{confidence:.2f}%"
    )

    # Probability for each species
    st.subheader("Prediction Probabilities")

    probabilities = pd.DataFrame({
        "Species": encoder.classes_,
        "Probability": prediction[0] * 100
    })

    st.bar_chart(
        probabilities.set_index("Species")
    )

    st.write("Input Measurements:")

    st.dataframe(
        pd.DataFrame([input_data], columns=features),
        use_container_width=True
    )


st.divider()

st.caption("Built using Python, TensorFlow, Scikit-learn and Streamlit.")