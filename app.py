import os
import joblib
import gradio as gr
import numpy as np


# -----------------------------
# Load Model
# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "diabetes_prediction_model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}. "
        "Please upload diabetes_prediction_model.pkl to your repository."
    )

model = joblib.load(MODEL_PATH)

print("✅ Diabetes prediction model loaded successfully")


# -----------------------------
# Prediction Function
# -----------------------------

def predict_diabetes(pregnancies, glucose, insulin, bmi, age):

    input_data = np.array([
        [
            float(pregnancies),
            float(glucose),
            float(insulin),
            float(bmi),
            float(age)
        ]
    ])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        return "🟥 High Risk of Diabetes (Positive)"
    else:
        return "🟩 Low Risk of Diabetes (Negative)"


# -----------------------------
# Gradio Interface
# -----------------------------

interface = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(
            label="Pregnancies",
            value=0
        ),
        gr.Number(
            label="Glucose",
            value=120
        ),
        gr.Number(
            label="Insulin",
            value=100
        ),
        gr.Number(
            label="BMI",
            value=25
        ),
        gr.Number(
            label="Age",
            value=30
        )
    ],
    outputs=gr.Textbox(
        label="Prediction Result"
    ),
    title="Diabetes Prediction System",
    description=(
        "Machine Learning diabetes risk prediction "
        "using Decision Tree Classifier."
    ),
    theme=gr.themes.Soft()
)


# -----------------------------
# Run Application
# -----------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 7860))

    interface.launch(
        server_name="0.0.0.0",
        server_port=port
    )
