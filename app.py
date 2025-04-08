from flask import Flask, request, jsonify, render_template, session
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv
import os
import google.generativeai as genai

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")  # or "gemini-pro"

# Load trained model and encoder
with open('disease_model_rf.pkl', 'rb') as model_file:
    model_rf = pickle.load(model_file)

with open('label_encoder_rf.pkl', 'rb') as le_file:
    label_encoder = pickle.load(le_file)

# Load data
disease_info_df = pd.read_csv('disease_info.csv')
precautions_df = pd.read_csv('disease_precautions.csv')
final_df = pd.read_csv('final.csv')
symptom_columns = [col for col in final_df.columns if col != 'Disease']

# Initialize Flask
app = Flask(__name__)
app.secret_key = 'b3f8d74901a92b3fcb58'  # For session

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    symptoms = {col: 0 for col in symptom_columns}
    input_symptoms = input_data.get('symptoms', [])

    valid_symptoms = [sym for sym in input_symptoms if sym in symptoms]
    for sym in valid_symptoms:
        symptoms[sym] = 1

    input_df = pd.DataFrame([symptoms])
    input_df = input_df[symptom_columns]

    predicted_encoded = model_rf.predict(input_df)
    predicted_disease = label_encoder.inverse_transform(predicted_encoded)

    probabilities = model_rf.predict_proba(input_df)
    confidence_scores = {
        disease: float(prob) for disease, prob in zip(label_encoder.classes_, probabilities[0])
    }

    # Store disease in session
    session['context'] = {'disease': predicted_disease[0]}

    return jsonify({
        'predicted_disease': predicted_disease[0],
        'confidence_scores': confidence_scores
    })

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    context = session.get('context', {})

    if 'disease' in context:
        disease = context['disease']
        prompt = f"""
        The user has been diagnosed with '{disease}'.
        They asked: "{user_message}"

        Please give a helpful and medically relevant answer considering the disease.
        Keep the tone informative but friendly.
        """
        try:
            response = model.generate_content(prompt)
            answer = response.text.strip()
        except Exception as e:
            answer = f"An error occurred while generating a response: {e}"
    else:
        answer = "Please provide symptoms first for disease prediction."

    session['context'] = {
        'disease': context.get('disease', None)
    }

    return jsonify({'response': answer})


if __name__ == '__main__':
    app.run(debug=True)
