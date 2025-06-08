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
    return "🔴 Liver Disease Detected" if prediction == 1 else "🟢 No Liver Disease Detected"

with gr.Blocks(title = "Liver Disease Predictor", theme = gr.themes.Soft()) as interface:
    gr.Markdown("🩺 Liver Disease Predictor")
    gr.Markdown("🔍 Enter patient diagnostic values below to predict liver disease risk.")

    with gr.Row():
        with gr.Column():
            age = gr.Number(label = "Age")
            gender = gr.Radio(["0", "1"], label="Gender (0: Female, 1: Male)")
            total_bilirubin = gr.Number(label="Total Bilirubin")
            direct_bilirubin = gr.Number(label="Direct Bilirubin")
            alk_phosphate = gr.Number(label="Alkaline Phosphotase")
            alamine = gr.Number(label="Alamine_Aminotransferase")
            aspartate = gr.Number(label="Aspartate_Aminotrasferase")
            proteins = gr.Number(label="Total_Proteins")
            albumin = gr.Number(label="Albumin")
            ag_ratio = gr.Number(label="Albumin_Globulin_Ratio")
        
        with gr.Column():
            result = gr.Testbox(label = "Prediction Result")
            submit_btn = gr.Button("🔎 Predict Now")
    submit_btn.click(
        fn = predict_liver_disease,
        inputs=[age, gender,total_bilirubin, direct_bilirubin,alk_phosphate, alamine,aspartate,proteins,albumin, ag_ratio],
        outputs= result
        )
    

if __name__ == "__main__":
    interface.launch()