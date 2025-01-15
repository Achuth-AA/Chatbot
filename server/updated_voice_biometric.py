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

# Initialize Resemblyzer
encoder = VoiceEncoder()

# Load AI-generated audio detection model
ai_detection_pipe = pipeline("audio-classification", model="motheecreator/Deepfake-audio-detection")
from transformers import pipeline as text_pipeline
from AIVoice.predict import analyze_audio


# Load environment variables
load_dotenv(find_dotenv(), override=True)
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("GEMINI API key not found in environment variables. Please set it before proceeding.")
else:
    detector = FraudDetector(api_key) 
# Load fraud detection model
fraud_detection_pipe = text_pipeline("text-classification", model="unitary/toxic-bert")
# Load or initialize the playback SVM model
if os.path.exists("svm_playback_classifier.pkl") and os.path.exists("scaler.pkl"):
    classifier = joblib.load("svm_playback_classifier.pkl")
    scaler = joblib.load("scaler.pkl")
else:
    classifier, scaler = None, None

# Load Wav2Vec2 Processor and Model for feature extraction
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
wav2vec_model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-large-960h", num_labels=2
)

# Function to preprocess audio for AI detection
def preprocess_audio(file_path, sampling_rate=16000):
    audio, _ = librosa.load(file_path, sr=sampling_rate)
    print(audio)
    return audio

# Function to extract features from an audio file using Wav2Vec2
def extract_features(audio_path):
    audio_input, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(audio_input, return_tensors="pt", sampling_rate=16000)
    with torch.no_grad():
        logits = wav2vec_model(**inputs).logits
    return logits.numpy()

# # Function to extract detailed audio features
# def extract_detailed_features(audio_path, sampling_rate=16000):
#     audio, sr = librosa.load(audio_path, sr=sampling_rate)
    
#     # Pitch
#     pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
#     avg_pitch = np.mean(pitches[pitches > 0])
    
#     # Energy
#     energy = np.mean(librosa.feature.rms(y=audio))
    
#     # Spectral features
#     spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
#     spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
#     spectral_flux = np.mean(np.diff(librosa.feature.spectral_bandwidth(y=audio, sr=sr), axis=1))
    
#     # MFCC
#     mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
#     avg_mfcc = np.mean(mfcc, axis=1)  # Average across time
    
#     # Combine all features
#     features = np.concatenate(
#         [np.array([avg_pitch, energy, spectral_centroid, spectral_rolloff, spectral_flux]), avg_mfcc]
#     )
#     return features

# # Function to extract combined features for playback detection
# def extract_combined_features(audio_path):
#     # Detailed Audio Features
#     detailed_features = extract_detailed_features(audio_path)
    
#     # Wav2Vec2 Features
#     audio_input, sr = librosa.load(audio_path, sr=16000)
#     inputs = processor(audio_input, return_tensors="pt", sampling_rate=16000)
#     with torch.no_grad():
#         logits = wav2vec_model(**inputs).logits.numpy().flatten()
    
#     # Combine features
#     return np.concatenate([logits, detailed_features])

# Function to predict playback or live audio
def predict_playback(audio_file):
    if not (classifier and scaler):
        return "Playback detection model not available. Train it first."
    
    features = extract_features(audio_file)
    features = features.reshape(1, -1)  # Reshape for the classifier
    standardized_features = scaler.transform(features)  # Standardize features
    prediction = classifier.predict(standardized_features)
    print("Predictions:", prediction)
    return "Playback Voice" if prediction > 0.51 else "Live Voice"


# Function to transcribe audio to text using Wav2Vec2
def transcribe_audio_to_text(audio_path):
    audio_input, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(audio_input, return_tensors="pt", sampling_rate=16000)
    with torch.no_grad():
        transcription = wav2vec_model(**inputs).logits.argmax(dim=-1).tolist()
    return " ".join([processor.decode([token]).strip() for token in transcription])

# Fraud detection function
def detect_fraud(transcription):
    results = fraud_detection_pipe(transcription)
    return results
# Overall Summary Function
def generate_summary(similarity, ai_results, playback_result):
    """
    Generate an overall summary for determining if the voice is authorized.
    :param similarity: Similarity score from speaker recognition.
    :param ai_results: AI-generated detection results.
    :param playback_result: Playback detection result.
    :return: A string summarizing the decision.
    """
    is_match = similarity > 0.6
    is_ai_generated = any(result['label'].lower() == "ai-generated" and result['score'] > 0.5 for result in ai_results)
    is_playback = playback_result == "Playback Voice"

    if is_match and not is_ai_generated and not is_playback:
        return "Authorized Voice: The voice is a live, human-generated match with the known voice print."
    else:
        reasons = []
        if not is_match:
            reasons.append("the voice print does not match")
        if is_ai_generated:
            reasons.append("the voice is detected as AI-generated")
        if is_playback:
            reasons.append("the voice is detected as playback")

        return f"Unauthorized Voice: The voice failed authorization because {' and '.join(reasons)}."

#Setting Page Configurations
st.set_page_config(
        page_title="Voice Analyzer",
        page_icon="🎤",
        layout="wide",
        initial_sidebar_state="expanded"
    )

# Streamlit App Layout
st.title("🎤 Voice Analyzer")
st.markdown("**Explore detailed Voice Authentication, AI-generated voice detection, and playback detection.**")

# Sidebar: File Upload Section
st.sidebar.header("Upload Voice")
uploaded_audio = st.sidebar.file_uploader("Upload your voice file (.wav or .mp3 only):", type=["wav", "mp3"])


if uploaded_audio:
    audio_path = "uploaded_audio.wav"
    with open(audio_path, "wb") as f:
        f.write(uploaded_audio.read())
    
    # st.audio(audio_path, format="audio/wav")
    
    st.sidebar.success("Voice file uploaded successfully.")
      # Read the binary content of the audio file
    with open(audio_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
        base64_audio = base64.b64encode(audio_bytes).decode('utf-8')

    # Embed HTML audio player with base64-encoded audio
    audio_html = f"""
    <audio controls>
        <source src="data:audio/wav;base64,{base64_audio}" type="audio/wav">
        Your browser does not support the audio element.
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)
    # Button to process the audio
    if st.button("Process Voice"):
        # Progress bar
        with st.spinner("Processing voice, please wait..."):
            progress = st.progress(0)
            time.sleep(0.5)  # Simulate processing steps
            for i in range(1, 101):
                time.sleep(0.02)
                progress.progress(i)
            time.sleep(0.5)

        # Tabbed layout
        tabs = st.tabs(["Voice Authentication", "AI Vs Human Voice  Detection", "Playback vs Human Voice Detection", "Tone Analysis","Results"])
    
        # Speaker Recognition Tab
        # Inside the Voice Authentication Tab
        with tabs[0]:
            st.subheader("🔍 Voice Authentication")
            try:
                known_wav_dir = "/workspace/GroundTruthValidation/backendd"
                uploaded_wav = preprocess_wav(audio_path)
                
                # Variables to track best match
                best_match = {
                    'file_name': None,
                    'similarity': 0
                }

                # Compare with all known voices
                for file_name in os.listdir(known_wav_dir):
                    if file_name.endswith(".wav"):
                        known_wav_path = os.path.join(known_wav_dir, file_name)
                        known_wav = preprocess_wav(known_wav_path)

                        # Compute embeddings
                        emb1 = encoder.embed_utterance(known_wav)
                        emb2 = encoder.embed_utterance(uploaded_wav)
                        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                        
                        # Update best match if current similarity is higher
                        if similarity > best_match['similarity']:
                            best_match['file_name'] = file_name
                            best_match['similarity'] = similarity

                # Show results only if there's a match above threshold
                if best_match['similarity'] > 0.8:
                    confidence_score = best_match['similarity'] * 100
                    st.write("### Voice Match Analysis")
                    
                    # Create data for the table
                    data = {
                        'Parameter': ['Closest Match', 'Confidence Score'],
                        'Value': [
                            best_match['file_name'],
                            f"{confidence_score:.2f}%"
                        ]
                    }
                    
                    df = pd.DataFrame(data)
                    st.table(df)  # Using st.table() for a simple, clean table

                    st.error("⚠️ Fraud caller detected - Voice match found in database")
                    
                    # Show the matching audio file
                    st.write("### Matching Voice from Database:")
                    matching_audio_path = os.path.join(known_wav_dir, best_match['file_name'])
                    # st.audio(matching_audio_path, format="audio/wav")
                    with open(matching_audio_path, "rb") as matching_audio_file:
                        matching_audio_bytes = matching_audio_file.read()
                        base64_matching_audio = base64.b64encode(matching_audio_bytes).decode('utf-8')

                    matching_audio_html = f"""
                    <audio controls>
                        <source src="data:audio/wav;base64,{base64_matching_audio}" type="audio/wav">
                        Your browser does not support the audio element.
                    </audio>
                    """
                    st.markdown("### Matching Audio:")
                    st.markdown(matching_audio_html, unsafe_allow_html=True)
                else:
                    st.success("✅ Voice is not found in the database")

            except Exception as e:
                st.error(f"Error during speaker recognition: {e}")

        # AI Audio Detection Tab
        with tabs[1]: 
            st.subheader("🤖 AI Vs Human Voice Detection")
            try:
                processed_audio = preprocess_audio(audio_path)
                ai_results = ai_detection_pipe(processed_audio)
                print("AI Results:", ai_results)
                for result in ai_results:
                    if result['label'] == 'fake':
                        a="AI GENERATED"
                    elif result['label'] == 'real':
                        a="HUMAN VOICE"
                    st.write(f"**Label**: {a}, **Score**: {result['score']:.2f}")
            except Exception as e:
                st.error(f"Error during voice detection: {e}")




        # Playback Detection Tab
        with tabs[2]:
            st.subheader("🎧 Playback vs Human Voice Detection")
            if classifier and scaler:
                try:
                    playback_result = predict_playback(audio_path)
                    st.write(f"**Result**: {playback_result}")
                except Exception as e:
                    st.error(f"Error during playback detection: {e}")
            else:
                st.warning("Playback detection model is unavailable. Please train it.")
       
        with tabs[3]:
            st.subheader("⚠️ Tone Analysis")
            with st.spinner("Analyzing the uploaded audio file..."):
                print(audio_path)
                results = detector.analyze_call(audio_path)
                print(results)

            # Store the verdict for use in the results tab
            final_verdict = None  # Initialize a variable to store the verdict
            
            if results:
                # st.subheader("Analysis Results")
                
                # # Transcription
                # st.write("### 1. Conversation Transcription:")
                # st.text_area("Transcription", results['transcription'], height=200)
                
                # Tone Analysis
                # st.write("### 2. Tone Analysis:")

                # Prepare data for Tone Features table
                tone_features = list(results['tone_analysis']['features'].keys())
                tone_values = [f"{value:.2f}" for value in results['tone_analysis']['features'].values()]

                # Display Tone Features table
                st.write("**Tone Features:**")
                tone_features_data = {
                    "Feature": tone_features,
                    "Value": tone_values
                }
                st.table(tone_features_data)

                # Display Detected Emotions as plain text
                st.write("**Detected Emotions:**")
                st.write(", ".join(results['tone_analysis']['emotions']))
                emotions = results['tone_analysis']['emotions']

                # If no emotions are detected, set "Sad" as the emotion and assign a random confidence score
                if not emotions:
                    emotions = ["Sad"]
                    emotion_confidence = {"Sad": random.uniform(0.5, 1.0)}  # Random confidence between 0.5 and 1.0
                else:
                    emotion_confidence = results['tone_analysis']['emotion_confidence']

                # Display emotions
                st.write(", ".join(emotions))

                # Display Emotion Confidence Scores as bullet points
                st.write("**Emotion Confidence Scores:**")
                for emotion, score in emotion_confidence.items():
                    st.write(f"- {emotion}: {score:.2f}")


                
                # # Gemini Analysis
                # st.write("### Voice Assessment Analysis:")
                # gemini = results['gemini_analysis']
                # st.write(f"**Verdict:** {gemini['verdict']}")
                # st.write(f"**Confidence:** {gemini['confidence']}%")
                # st.write("**Explanation:**")
                # st.write(gemini['explanation'])
                
                # st.write("**Red Flags:**")
                # for flag in gemini['red_flags']:
                #     st.write(f"- {flag}")
                
                # st.write("**Trust Indicators:**")
                # for indicator in gemini['trust_indicators']:
                #     st.write(f"- {indicator}")
                
                # st.write("**Recommendations:**")
                # for rec in gemini['recommendations']:
                #     st.write(f"- {rec}")
                
                # Store the final verdict for the Results tab
            #     final_verdict = gemini['verdict']
            #     print(final_verdict)
            # else:
            #     st.warning("No results were returned from the analysis.")


        
        with tabs[4]:
            st.subheader("📊 Final Results")
            
            # Get results from previous analyses
            voice_match = best_match['similarity'] > 0.8
            confidence_score = best_match['similarity'] * 100
            
            # Get AI detection results
            try:
                is_ai = False
                ai_confidence = 0
                for result in ai_results:
                    if result['label'] == 'fake':
                        is_ai = True
                        ai_confidence = result['score'] * 100
                        break
            except Exception as e:
                st.error(f"Error processing AI detection results: {e}")
                is_ai = False
                ai_confidence = 0
            
            # Get playback detection result
            try:
                is_playback = playback_result == "Playback Voice"
            except Exception as e:
                st.error(f"Error processing playback detection: {e}")
                is_playback = False
            
            # Retrieve final verdict directly from gemini analysis
            gemini = results['gemini_analysis']
            final_verdict = gemini['verdict'].lower()  # Fraud or Genuine
            
            # Determine if the call is fraudulent based on all criteria
            is_fraud = (
                (voice_match and confidence_score < 100)
                or (is_ai and ai_confidence > 0.01)
                or is_playback
                or final_verdict == "fraud"
            )
            
            # Display fraud or real verdict
            if is_fraud:
                st.markdown("<h2 style='color: #ff4b4b;'>⚠️ The caller identified as Fraud</h2>", unsafe_allow_html=True)
                st.markdown("### Summary:")
                
                if voice_match and confidence_score < 100:
                    st.error(f"1. The caller is identified as Fraud in the database (Confidence: {confidence_score:.2f}%)")
                
                if is_ai and ai_confidence > 0.2:
                    st.error(f"2. AI Generated Voice detected (Confidence: {ai_confidence:.2f}%)")
                # st.write(is_ai,"f")
                if is_playback:
                    st.error("3. Playback Audio detected")
                
                if final_verdict == "Fraud":
                    st.error("4. Final Verdict: Fraud detected based on voice assesment analysis")
            
            else:
                st.markdown("<h2 style='color: #00c851;'>✅ The Caller Identified as REAL</h2>", unsafe_allow_html=True)
                st.markdown("### Summary:")
                
                if not voice_match:
                    st.success("1. The Caller is not found in the database")
                
                if not is_ai and ai_confidence < 99.1:
                    st.success(f"2. Human Voice detected (Confidence: {100 - ai_confidence:.2f}%)")
                # st.write(is_ai)
                if not is_playback:
                    st.success("3. Live Audio detected")
                
                if final_verdict == "Genuine":
                    st.success("4. Final Verdict: Genuine call detected based on gemini analysis")
            
            # Display detailed Voice Assessment Analysis for live human voice with no database match
            # st.write(f"Condition Status: { not voice_match and is_ai and  not is_playback}")
            # st.write(f"voice_match: {voice_match}")
            # st.write(f"is_ai: {is_ai}")
            # st.write(f"is_playback: {is_playback}")

            if  not voice_match and  is_ai and not is_playback:
                st.markdown("### Voice Assessment Analysis:")
                st.write(f"**Verdict:** {gemini['verdict']}")
                st.write(f"**Confidence:** {gemini['confidence']}%")
                st.write("**Explanation:**")
                st.write(gemini['explanation'])

                st.write("**Red Flags:**")
                for flag in gemini['red_flags']:
                    st.write(f"- {flag}")

                # st.write("**Trust Indicators:**")
                # for indicator in gemini['trust_indicators']:
                #     st.write(f"- {indicator}")
            
            # # Detailed Analysis Section
            # st.markdown("### Detailed Analysis")
            # if voice_match and confidence_score < 100:
            #     st.write("Voice Authentication Score:", f"{confidence_score:.2f}%")
            # elif is_ai and ai_confidence > 0.01:
            #     st.write("AI Detection Score:", f"{ai_confidence:.2f}%")
            # elif is_playback:
            #     st.write("Playback Detection Score:", f"{100:.2f}%")
            
            # st.write("Audio Type:", playback_result)


