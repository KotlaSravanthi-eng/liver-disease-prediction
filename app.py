import gradio as gr
import numpy as np
import pickle

# load the trained model
model_path = 'artifacts/model_trainer.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Define prediction function 
def predict_liver_disease(age, gender, total_bilirubin, direct_bilirubin, alk_phosphate, alamine, aspartate, proteins, albumin, ag_ratio):
    input_data = np.array([[age, gender,total_bilirubin, direct_bilirubin,alk_phosphate, alamine,aspartate,proteins,albumin, ag_ratio]])

    prediction = model.predict(input_data)[0]
    return "Liver Disease Detected" if prediction == 1 else "No Liver Disease"

# Gradio interface
iface = gr.Interface(
    fn = predict_liver_disease,
    inputs = [
        gr.Number(label = "Age"),
        gr.Radio(["0", "1"], label="Gender (0: Female, 1: Male)"),
        gr.Number(label="Total Bilirubin"),
        gr.Number(label="Direct Bilirubin"),
        gr.Number(label="Alkaline Phosphotase"),
        gr.Number(label="Alamine_Aminotransferase"),
        gr.Number(label="Aspartate_Aminotrasferase"),
        gr.Number(label="Total_Proteins"),
        gr.Number(label="Albumin"),
        gr.Number(label="Albumin_Globulin_Ratio")
    ],
    outputs= gr.Text(label="Prediction"),
    title="Liver Disease Prediction App"
)

if __name__ == "__main__":
    iface.launch()