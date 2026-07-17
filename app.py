import os
import joblib
import gradio as gr
import numpy as np

# Load the trained Decision Tree model
model = joblib.load("diabetes_prediction_model.pkl")


def predict_diabetes(pregnancies, glucose, insulin, bmi, age):
    """
    Predict diabetes risk using the trained Decision Tree model.
    The feature order must be the same as used during training.
    """

    input_data = np.array([
        [pregnancies, glucose, insulin, bmi, age]
    ])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        return "🟥 High Risk of Diabetes (Positive)"
    else:
        return "🟩 Low Risk of Diabetes (Negative)"


# Create Gradio Interface
interface = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(label="Pregnancies"),
        gr.Number(label="Glucose"),
        gr.Number(label="Insulin"),
        gr.Number(label="BMI"),
        gr.Number(label="Age")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Diabetes Prediction System",
    description="Enter the patient's medical information to predict diabetes risk using a Decision Tree model.",
    theme=gr.themes.Soft()
)


if __name__ == "__main__":
    interface.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
