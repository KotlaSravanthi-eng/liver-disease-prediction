import gradio as gr

from src.pipelines.predict_pipeline import CustomData, PredictPipeline
from src.utils import load_object

# load the trained model
model = load_object('artifacts/model_trainer.pkl')
preprocessor = load_object('artifacts/preprocessor.pkl')

# Define prediction function 
def predict_liver_disease(age, gender, total_bilirubin, direct_bilirubin, alk_phosphate, alamine, aspartate, total_protiens, albumin, ag_ratio):
    try:
        user_data = CustomData(
            age=age,
            gender=gender,
            total_bilirubin=total_bilirubin,
            direct_bilirubin=direct_bilirubin,
            alkphos = alk_phosphate,
            alamine= alamine,
            aspartate=aspartate,
            total_protiens=total_protiens,
            albumin=albumin,
            ag_ratio=ag_ratio
        )
        input_data = user_data.get_data_as_data_frame()

        prediction_pipeline = PredictPipeline()
        prediction = prediction_pipeline.predict(input_data)[0]

        return "🔴 Liver Disease Detected" if prediction == 1 else "🟢 No Liver Disease Detected"
    except Exception as e:
        return f"Error :{str(e)}"
    
with gr.Blocks(title = "Liver Disease Predictor", theme = gr.themes.Soft()) as interface:
    gr.Markdown("🩺 **LIVER DISEASE PREDICTOR**")
    gr.Markdown("🔍 Enter patient diagnostic values below to predict liver disease risk.")

    with gr.Row():
        with gr.Column():
            age = gr.Number(label = "Age", value=45, minimum=1,maximum=100,step=1)
            gender = gr.Radio(["Female", "Male"], label="Gender")
            total_bilirubin = gr.Number(label="Total Bilirubin", value=1.0, minimum=0.0, maximum=10.0, step=0.1)
            direct_bilirubin = gr.Number(label="Direct Bilirubin", value=0.5, minimum=0.0, maximum=5.0, step=0.1)
            alk_phosphate = gr.Number(label="Alkaline Phosphotase", value=200, minimum=50, maximum=300, step=1)
            alamine = gr.Number(label="Alamine Aminotransferase", value=30, minimum=0, maximum=300, step=1)
            aspartate = gr.Number(label="Aspartate Aminotransferase", value=35, minimum=0, maximum=300, step=1)
            total_protiens = gr.Number(label="Total Protiens", value=6.5, minimum=0.0, maximum=10.0, step=0.1)
            albumin = gr.Number(label="Albumin", value=3.5, minimum=0.0, maximum=9.0, step=0.1)
            ag_ratio = gr.Number(label="Albumin Globulin Ratio", value=1.0, minimum=0.0, maximum=3.0, step=0.1)
        
        with gr.Column():
            result = gr.Textbox(label = "Prediction Result")
            submit_btn = gr.Button("🔎 Predict Now")
    submit_btn.click(
        fn = predict_liver_disease,
        inputs=[age, gender,total_bilirubin, direct_bilirubin,alk_phosphate, alamine,aspartate,total_protiens,albumin, ag_ratio],
        outputs= result
        )
    

if __name__ == "__main__":
    interface.launch(share = True)