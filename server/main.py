from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import librosa
from resemblyzer import VoiceEncoder, preprocess_wav
import os
import csv
from transformers import pipeline
import joblib
from fraud_detector import FraudDetector
from tone_analyzer import ToneAnalyzer
from werkzeug.utils import secure_filename
import time
from dotenv import load_dotenv
import torch
import json
import uuid
import os
import librosa
import librosa.display
import numpy as np
import base64
from playsound import playsound
import pandas as pd
import random 
from resemblyzer import VoiceEncoder, preprocess_wav
import streamlit as st
from transformers import pipeline, Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
import torch
import time
from fraud_detector import FraudDetector
from dotenv import load_dotenv, find_dotenv
import os
from pathlib import Path
from speech2text import run_analysis
import traceback


app = Flask(__name__)
CORS(app)

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Initialize all required components
encoder = VoiceEncoder()
detector = FraudDetector(api_key)
tone_analyzer = ToneAnalyzer()
ai_detection_pipe = pipeline("audio-classification", model="motheecreator/Deepfake-audio-detection")

# Load SVM model and scaler
classifier = joblib.load("svm_playback_classifier.pkl")
scaler = joblib.load("scaler.pkl")

# Configure folders
UPLOAD_FOLDER = 'temp_uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# CSV file path
CSV_FILE = os.path.join(RESULTS_FOLDER, 'analysis_results.csv')
DETAILED_RESULTS_FOLDER = os.path.join(RESULTS_FOLDER, 'detailed')
os.makedirs(DETAILED_RESULTS_FOLDER, exist_ok=True)



processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
wav2vec_model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-large-960h", num_labels=2
)

# Initialize CSV if it doesn't exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'timestamp', 'filename', 'final_verdict', 'detailed_result_file'])

def extract_features(audio_path):
    audio_input, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(audio_input, return_tensors="pt", sampling_rate=16000)
    with torch.no_grad():
        logits = wav2vec_model(**inputs).logits
    return logits.numpy()

def process_voice_authentication(audio_path):
    try:
        known_wav_dir = "/workspace/Chatbot/server/backenddd"  # Your database directory
        uploaded_wav = preprocess_wav(audio_path)
        best_match = {'file_name': None, 'similarity': 0}

        for file_name in os.listdir(known_wav_dir):
            if file_name.endswith(".wav"):
                known_wav_path = os.path.join(known_wav_dir, file_name)
                known_wav = preprocess_wav(known_wav_path)
                emb1 = encoder.embed_utterance(known_wav)
                emb2 = encoder.embed_utterance(uploaded_wav)
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                
                if similarity > best_match['similarity']:
                    best_match['file_name'] = file_name
                    best_match['similarity'] = similarity

        return {
            "matched": best_match['similarity'] > 0.8,
            "match_confidence": float(best_match['similarity'] * 100),
            "matched_file": best_match['file_name'] if best_match['similarity'] > 0.8 else None
        }
    except Exception as e:
        print(f"Error in voice authentication: {str(e)}")
        return {"error": str(e)}

def process_ai_detection(audio_path):
    try:
        audio, _ = librosa.load(audio_path, sr=16000)
        results = ai_detection_pipe(audio)
        return {
            "is_ai": any(r['label'] == 'fake' and r['score'] > 0.5 for r in results),
            "detailed_results": results
        }
    except Exception as e:
        print(f"Error in AI detection: {str(e)}")
        return {"error": str(e)}

def process_playback_detection(audio_path):
    try:
        # Extract features (you need to implement this based on your model)
        features = extract_features(audio_path)  # Implement this function
        features = features.reshape(1, -1)
        standardized_features = scaler.transform(features)
        prediction = classifier.predict(standardized_features)
        
        return {
            "is_playback": bool(prediction > 0.51),
            "confidence": float(prediction[0] if isinstance(prediction, np.ndarray) else prediction)
        }
    except Exception as e:
        print(f"Error in playback detection: {str(e)}")
        return {"error": str(e)}

def manage_results(analysis_result, result_id):
    try:
        csv_file = 'call_analysis_results.csv'
        json_file = f'detailed_results/{result_id}.json'
        os.makedirs('detailed_results', exist_ok=True)

        # Store detailed JSON
        with open(json_file, 'w') as f:
            json.dump(analysis_result, f, indent=4)

        # Prepare CSV row
        csv_row = {
            'ID': result_id,
            'Timestamp': analysis_result['timestamp'],
            'Filename': analysis_result['filename'],
            'Final Verdict': analysis_result['final_verdict'],
            'Voice Match': analysis_result['voice_authentication']['matched'],
            'AI Detection': analysis_result['ai_detection']['is_ai'],
            'Playback Detection': analysis_result['playback_detection']['is_playback'],
            'Confidence Score': analysis_result['detailed_analysis']['gemini_analysis']['confidence'],
            'JSON File': json_file
        }

        # Create/append to CSV
        file_exists = os.path.exists(csv_file)
        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(csv_row)

        return True
    except Exception as e:
        print(f"Error managing results: {str(e)}")
        return False

@app.route('/process', methods=['POST'])
def process_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        result_id = str(uuid.uuid4())
        
        auth_results = process_voice_authentication(filepath)
        ai_results = process_ai_detection(filepath)
        playback_results = process_playback_detection(filepath)
        detector_results = detector.analyze_call(filepath)

        is_fraud = (
            auth_results.get('matched', False) or
            ai_results.get('is_ai', False) or
            playback_results.get('is_playback', False) or
            (detector_results and detector_results.get('gemini_analysis', {}).get('verdict', '').lower() == 'fraud')
        )

        analysis_result = {
            "id": result_id,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "filename": filename,
            "final_verdict": "FRAUD" if is_fraud else "GENUINE",
            "voice_authentication": {
                **auth_results,
                "matched": str(auth_results.get('matched', False)).lower()
            },
            "ai_detection": {
                **ai_results,
                "is_ai": str(ai_results.get('is_ai', False)).lower()
            },
            "playback_detection": {
                **playback_results,
                "is_playback": str(playback_results.get('is_playback', False)).lower()
            },
            "detailed_analysis": detector_results
        }

        manage_results(analysis_result, result_id)

        if os.path.exists(filepath):
            os.remove(filepath)

        return jsonify({
            "success": True,
            "result": analysis_result
        })

    except Exception as e:
        print(e)
        traceback.print_exc()
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500



# @app.route('/process', methods=['POST'])
# def process_audio():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file provided'}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'}), 400

#     try:
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(UPLOAD_FOLDER, filename)
#         file.save(filepath)

#         result_id = str(uuid.uuid4())
        
#         # Process all analyses
#         auth_results = process_voice_authentication(filepath)
#         ai_results = process_ai_detection(filepath)
#         playback_results = process_playback_detection(filepath)
#         detector_results = detector.analyze_call(filepath)

#         # Convert boolean values to strings for JSON serialization
#         is_fraud = (
#             auth_results.get('matched', False) or
#             ai_results.get('is_ai', False) or
#             playback_results.get('is_playback', False) or
#             (detector_results and detector_results.get('gemini_analysis', {}).get('verdict', '').lower() == 'fraud')
#         )

#         timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
#         detailed_result_file = f"{result_id}.json"
#         detailed_result_path = os.path.join(DETAILED_RESULTS_FOLDER, detailed_result_file)

#         # Ensure all boolean values are converted to strings
#         analysis_result = {
#             "id": result_id,
#             "timestamp": timestamp,
#             "filename": filename,
#             "final_verdict": "FRAUD" if is_fraud else "GENUINE",
#             "voice_authentication": {
#                 **auth_results,
#                 "matched": str(auth_results.get('matched', False)).lower()
#             },
#             "ai_detection": {
#                 **ai_results,
#                 "is_ai": str(ai_results.get('is_ai', False)).lower()
#             },
#             "playback_detection": {
#                 **playback_results,
#                 "is_playback": str(playback_results.get('is_playback', False)).lower()
#             },
#             "detailed_analysis": detector_results
#         }

#         with open(detailed_result_path, 'w') as f:
#             json.dump(analysis_result, f, indent=4)

#         with open(CSV_FILE, 'a', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow([result_id, timestamp, filename, "FRAUD" if is_fraud else "GENUINE", detailed_result_file])

#         if os.path.exists(filepath):
#             os.remove(filepath)

#         return jsonify({
#             "success": True,
#             "result": analysis_result
#         })

#     except Exception as e:
#         print(e)
#         traceback.print_exc()

#         if os.path.exists(filepath):
#             os.remove(filepath)
#         return jsonify({
#             "success": False,
#             "error": str(e)
#         }), 500

@app.route('/results', methods=['GET'])
def get_results():
    try:
        results = []
        with open(CSV_FILE, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Read detailed results
                detailed_file = os.path.join(DETAILED_RESULTS_FOLDER, row['detailed_result_file'])
                if os.path.exists(detailed_file):
                    with open(detailed_file, 'r') as df:
                        detailed_results = json.load(df)
                        results.append(detailed_results)
                else:
                    results.append(row)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/results/<result_id>', methods=['GET'])
def get_result(result_id):
    try:
        detailed_file = os.path.join(DETAILED_RESULTS_FOLDER, f"{result_id}.json")
        if os.path.exists(detailed_file):
            with open(detailed_file, 'r') as f:
                result = json.load(f)
                return jsonify(result)
        return jsonify({"error": "Result not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)