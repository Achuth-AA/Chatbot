import librosa
import numpy as np
from scipy.stats import skew, kurtosis
import python_speech_features

class ToneAnalyzer:
    def __init__(self):
        self.emotion_thresholds = {
            'angry': {'pitch_mean': 200, 'energy_mean': 0.7},
            'stressed': {'pitch_std': 50, 'energy_std': 0.3},
            'nervous': {'speech_rate': 180, 'pitch_range': 150},
            'suspicious': {'pause_ratio': 0.4, 'pitch_changes': 10}
        }

    def extract_tone_features(self, audio, sr):
        """Extract comprehensive tone features from audio"""
        try:
            # Basic Features
            pitch, magnitudes = librosa.piptrack(y=audio, sr=sr)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr)
            
            # Pitch Features
            pitches = pitch[magnitudes > 0]
            pitch_features = {
                'pitch_mean': np.mean(pitches) if len(pitches) > 0 else 0,
                'pitch_std': np.std(pitches) if len(pitches) > 0 else 0,
                'pitch_range': np.ptp(pitches) if len(pitches) > 0 else 0
            }

            # Energy Features
            energy = librosa.feature.rms(y=audio)[0]
            energy_features = {
                'energy_mean': np.mean(energy),
                'energy_std': np.std(energy),
                'energy_range': np.ptp(energy)
            }

            # Rhythm and Tempo Features
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
            rhythm_features = {
                'tempo': tempo,
                'onset_strength_mean': np.mean(onset_env),
                'onset_strength_std': np.std(onset_env)
            }

            # Speech Rate Estimation
            mfcc_delta = librosa.feature.delta(mfcc)
            speech_rate = np.mean(np.abs(mfcc_delta))

            # Pause Analysis
            intervals = librosa.effects.split(audio, top_db=20)
            pause_durations = np.diff([i[1] - i[0] for i in intervals]) / sr
            pause_features = {
                'pause_duration_mean': np.mean(pause_durations) if len(pause_durations) > 0 else 0,
                'pause_duration_std': np.std(pause_durations) if len(pause_durations) > 0 else 0,
                'pause_ratio': len(intervals) / (len(audio) / sr)
            }

            # Combine all features
            features = {
                **pitch_features,
                **energy_features,
                **rhythm_features,
                'speech_rate': float(speech_rate),
                **pause_features
            }

            return features

        except Exception as e:
            print(f"Error extracting tone features: {str(e)}")
            return None

    def detect_emotions(self, features):
        """Detect emotions based on tone features"""
        emotions = []
        confidence_scores = {}

        if not features:
            return emotions, confidence_scores

        # Check for anger
        if (features['pitch_mean'] > self.emotion_thresholds['angry']['pitch_mean'] and 
            features['energy_mean'] > self.emotion_thresholds['angry']['energy_mean']):
            emotions.append('angry')
            confidence_scores['angry'] = min(
                features['pitch_mean'] / self.emotion_thresholds['angry']['pitch_mean'],
                features['energy_mean'] / self.emotion_thresholds['angry']['energy_mean']
            )

        # Check for stress
        if (features['pitch_std'] > self.emotion_thresholds['stressed']['pitch_std'] and 
            features['energy_std'] > self.emotion_thresholds['stressed']['energy_std']):
            emotions.append('stressed')
            confidence_scores['stressed'] = min(
                features['pitch_std'] / self.emotion_thresholds['stressed']['pitch_std'],
                features['energy_std'] / self.emotion_thresholds['stressed']['energy_std']
            )

        # Check for nervousness
        if (features['speech_rate'] > self.emotion_thresholds['nervous']['speech_rate'] and 
            features['pitch_range'] > self.emotion_thresholds['nervous']['pitch_range']):
            emotions.append('nervous')
            confidence_scores['nervous'] = min(
                features['speech_rate'] / self.emotion_thresholds['nervous']['speech_rate'],
                features['pitch_range'] / self.emotion_thresholds['nervous']['pitch_range']
            )

        # Check for suspicious behavior
        if (features['pause_ratio'] > self.emotion_thresholds['suspicious']['pause_ratio']):
            emotions.append('suspicious')
            confidence_scores['suspicious'] = features['pause_ratio'] / self.emotion_thresholds['suspicious']['pause_ratio']

        return emotions, confidence_scores