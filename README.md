# 🩺 Liver Disease Prediction using Machine Learning

## 📊 Project Description
This project predicts the likelihood of liver disease based on medical diagnostic features using machine learning models.  
Built with Python, scikit-learn, and K-Nearest Neighbors, and deployed as an interactive web app using Gradio on Hugging Face Spaces.

After testing several models like XGBoost and CatBoost, KNN gave the best results.

--------------------

## 💡 Features
- Input features include Age, Total Bilirubin, Albumin, and other liver-related medical parameters  
- Predicts liver disease presence or absence  
- Clean and user-friendly interface  
- Easily accessible via Hugging Face Spaces  

## 🔍 Live Demo
Try the app here: [Liver Disease Predictor Demo](https://huggingface.co/spaces/kotlasravanthi/Liver-Disease-Predictor)

## 🎥 Demo

![Liver Disease Predictor Demo](C:\Users\Admin\OneDrive\Desktop\liver-disease-prediction\demo\live_demo_sample.gif)


-------

## 🧪 Sample Prediction

**Input Example:**
- Age: 60  
- Gender: Male  
- Total Bilirubin: 1.5  
- Direct Bilirubin: 0.5  
- Alkaline Phosphotase: 210  
- Alamine Aminotransferase: 30  
- Aspartate Aminotransferase: 40  
- Total Proteins: 6.8  
- Albumin: 3.2  
- Albumin and Globulin Ratio: 1.0  

**Output:**  
🟢 No Liver Disease (Prediction: 2)

---

## 🧠 Model Details
- Model: K-Nearest Neighbors (KNN) 
- Preprocessing: 
  - PowerTransformer for numerical features (in model pipeline)
  - Label Encoding for categorical column (used in EDA stage) 
- Handled class imbalance using `SMOTE` from `imbalanced-learn`
- F1 score: 85%  
- Dataset: [ILPD Indian Liver Patient Dataset](https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset)

## 📉 Confusion Matrix & ROC Curve
<p float="left"> 
<img src="C:\Users\Admin\OneDrive\Desktop\liver-disease-prediction\artifacts\confusion_matrix_K-Nearest_Neighbors.png" width="45%" /> 
<img src="C:\Users\Admin\OneDrive\Desktop\liver-disease-prediction\artifacts\roc_curve.png" width="45%" /> 
</p>

## 🛠️ Tech Stack
- Python  
- Pandas, NumPy  
- Matplotlib, seaborn
- Scikit-learn, XGBoost, CatBoost, KNN
- Imbalanced-learn (SMOTE)
- Dill (for saving objects)  
- Gradio (for UI)  
- Hugging Face Spaces (for deployment)


## 📁 Project Structure
├── artifacts/
|   ├── model_trainer.pkl # Trained ML model
|   └── preprocessor.pkl # preprocessed ML model
├── notebook/
|   ├── data/
|   |   ├── indian_liver_patient.csv 
|   ├── Liver_disease_prediction.ipynb
|   ├── Model_Training.ipynb
├── src/
│   ├── components/
│   │   ├── data_ingestion.py 
|   |   ├── data_transformation.py # Transformation of data
│   │   └── model_trainer.py # Training models
|   ├── pipelines/
|   |   ├── predict_pipeline.py # Prediction pipeline
|   |   └── train_pipeline.py # Training pipeline
│   ├── utils.py # Utility functions
│   ├── logger.py # logger for app
│   └── exception.py # Custom exceptions
├── requirements.txt # Required Python packages
├── app.py # app file
└── README.md # Project documentation


## 🧠 How to Run Locally
1. Clone the repo  
2. Install dependencies:
```bash
pip install -r requirements.txt


🤝 Contact
Created by Sravanthi Kotla
GitHub | https://github.com/KotlaSravanthi-eng

