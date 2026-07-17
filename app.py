import os
import joblib
import gradio as gr

# Load the trained Decision Tree model
MODEL_PATH = "diabetes_prediction_model.pkl"

try:
    deployed_dt = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise FileNotFoundError(
        f"Model file '{MODEL_PATH}' not found. Please place it in the same folder as app.py."
    )


# Prediction function
def predict_diabetes(pregnancies, glucose, insulin, bmi, age):
    """
    Predict diabetes risk using the trained Decision Tree model.
    """

    # Arrange inputs in the same order used during training
    input_data = [[
        float(pregnancies),
        float(glucose),
        float(insulin),
        float(bmi),
        float(age)
    ]]

    # Make prediction
    prediction = deployed_dt.predict(input_data)[0]

    # Return result
    if prediction == 1:
        return "🩺 Prediction: High Risk of Diabetes (Positive)"
    else:
        return "✅ Prediction: Low Risk of Diabetes (Negative)"


# Create Gradio Interface
interface = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(label="Pregnancies (Number of pregnancies)", value=0),
        gr.Number(label="Glucose (Plasma glucose concentration)", value=120),
        gr.Number(label="Insulin (2-Hour serum insulin)", value=80),
        gr.Number(label="BMI (Body Mass Index)", value=25.0),
        gr.Number(label="Age (Years)", value=30),
    ],
    outputs=gr.Textbox(label="Assessment Result"),
    title="Diabetes Prediction System",
    description=(
        "Enter the patient's medical measurements to predict the "
        "risk of diabetes using a trained Decision Tree Machine Learning model."
    ),
    theme=gr.themes.Soft(),
)


# Launch the application
if __name__ == "__main__":
    interface.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
