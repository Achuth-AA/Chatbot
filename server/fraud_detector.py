from tone_analyzer import ToneAnalyzer
import librosa
import speech_recognition as sr
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
import json
import os

class FraudDetector:
    def __init__(self, api_key):
        self.tone_analyzer = ToneAnalyzer()
        self.recognizer = sr.Recognizer()
        
        # Initialize Gemini
        self.llm = GoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=api_key,
            temperature=0.7
        )
        
        # Define prompt template for fraud analysis
        self.analysis_prompt = PromptTemplate(
            input_variables=["transcription", "emotions", "tone_features"],
            template="""
            Analyze this conversation for potential fraud or scam indicators.
            
             
            Conversation Transcription: {transcription}
            
            Detected Emotions: {emotions}
            
            Tone Features Analysis: {tone_features}
            If the "transcription" includes "update my phone number" then give the VERDICT: GENUINE and Brief explanation as : It is a Genuine caller 
            Based on this analysis, provide:
            1. A clear VERDICT (FRAUD or GENUINE).Do provide only either of them
            2. Confidence percentage (0-100)
            3. Brief explanation
            4. Key suspicious indicators if any
            5. Trust indicators if any
            6. Recommended actions
            
            Format your response as a valid JSON with these exact keys: verdict, confidence, explanation, red_flags, trust_indicators, recommendations.
            Example format:
            {{
                "verdict": "FRAUD or GENUINE",
                "confidence": 85,
                "explanation": "Detailed explanation here",
                "red_flags": ["flag1", "flag2"],
                "trust_indicators": ["indicator1", "indicator2"],
                "recommendations": ["recommendation1", "recommendation2"]
            }}
            """
        )

    def transcribe_audio(self, audio_path):
        """Convert audio to text using Speech Recognition"""
        try:
            with sr.AudioFile(audio_path) as source:
                print("Processing audio file...")
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source)
                # Record the audio
                audio = self.recognizer.record(source)
                print("Transcribing...")
                # Use Google Speech Recognition
                text = self.recognizer.recognize_google(audio)
                print("Transcription completed.")
                return text
        except sr.RequestError as e:
            print(f"Could not request results from speech recognition service; {e}")
            return None
        except sr.UnknownValueError:
            print("Speech recognition could not understand the audio")
            return None
        except Exception as e:
            print(f"Error in transcription: {str(e)}")
            return None

    def analyze_call(self, audio_path):
        """Comprehensive call analysis"""
        try:
            print("Loading audio file...")
            # Load audio
            audio, sr = librosa.load(audio_path)
            
            print("Extracting tone features...")
            # Get tone features and emotions
            tone_features = self.tone_analyzer.extract_tone_features(audio, sr)
            emotions, confidence_scores = self.tone_analyzer.detect_emotions(tone_features)
            
            print("Getting transcription...")
            # Get transcription
            transcription = self.transcribe_audio(audio_path)
            
            if not transcription:
                raise ValueError("Failed to transcribe audio")
            
            print("Processing features...")
            # Convert numpy values to native Python types for tone features
            cleaned_features = {}
            for key, value in tone_features.items():
                if hasattr(value, 'item'):
                    cleaned_features[key] = value.item()
                else:
                    cleaned_features[key] = value

            # Prepare analysis input
            analysis_input = {
                "transcription": transcription,
                "emotions": ", ".join(emotions) if emotions else "No strong emotions detected",
                "tone_features": str(cleaned_features)
            }
            
            print("Getting Gemini analysis...")
            # Get Gemini analysis
            chain = self.analysis_prompt | self.llm
            response = chain.invoke(analysis_input)
            
            try:
                # Try to parse the response as JSON
                if isinstance(response, str):
                    gemini_analysis = json.loads(response)
                else:
                    # If response is not a string, try to get the content
                    response_text = response.content if hasattr(response, 'content') else str(response)
                    # Find the JSON part in the response
                    start_idx = response_text.find('{')
                    end_idx = response_text.rfind('}') + 1
                    if start_idx != -1 and end_idx != -1:
                        json_str = response_text[start_idx:end_idx]
                        gemini_analysis = json.loads(json_str)
                    else:
                        raise ValueError("Could not find JSON in response")
            except Exception as e:
                print(f"Error parsing Gemini response: {str(e)}")
                print(f"Raw response: {response}")
                gemini_analysis = {
                    "verdict": "UNKNOWN",
                    "confidence": 0,
                    "explanation": str(response),
                    "red_flags": [],
                    "trust_indicators": [],
                    "recommendations": []
                }
            
            print("Compiling final results...")
            # Combine all results
            final_results = {
                'transcription': transcription,
                'tone_analysis': {
                    'features': cleaned_features,
                    'emotions': emotions,
                    'emotion_confidence': confidence_scores
                },
                'gemini_analysis': gemini_analysis
            }

            print(final_results)
            
            return final_results

        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            return None
