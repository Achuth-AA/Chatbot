from fraud_detector import FraudDetector
from dotenv import load_dotenv, find_dotenv
import os
from pathlib import Path

def analyze_audio_file(audio_file_path):
    """
    Analyze an audio file for fraud detection and tone analysis.
    
    Args:
        audio_file_path (str): Path to the audio file to analyze.
        
    Returns:
        dict: Analysis results, or None if analysis fails.
    """
    # Load environment variables
    load_dotenv(find_dotenv(), override=True)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI API key not found in environment variables.")
    
    # Initialize FraudDetector
    detector = FraudDetector(api_key)
    
    # Verify audio file exists
    audio_file = Path(audio_file_path)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found at {audio_file_path}")
    
    try:
        # Analyze call
        print("\nStarting analysis...")
        print("=" * 80)
        print(f"Analyzing file: {audio_file}")
        
        audio_path = str(audio_file.absolute()).replace('\\', '/')
        results = detector.analyze_call(audio_path)
        
        return results
    except Exception as e:
        raise RuntimeError(f"Error during analysis: {str(e)}")

def print_analysis_results(results):
    """Pretty print analysis results."""
    print("\nAnalysis Results:")
    print("=" * 80)
    
    # Print Transcription
    print("\n1. Conversation Transcription:")
    print("-" * 40)
    print(results['transcription'])
    
    # Print Tone Analysis
    print("\n2. Tone Analysis:")
    print("-" * 40)
    print("\nTone Features:")
    for feature, value in results['tone_analysis']['features'].items():
        if hasattr(value, 'item'):
            value = value.item()  # Convert numpy values to native Python types
        print(f"  - {feature}: {value:.2f}")
    
    print("\nDetected Emotions:", ", ".join(results['tone_analysis']['emotions']))
    print("\nEmotion Confidence Scores:")
    for emotion, score in results['tone_analysis']['emotion_confidence'].items():
        if hasattr(score, 'item'):
            score = score.item()
        print(f"  - {emotion}: {score:.2f}")
    
    # Print Gemini Analysis
    print("\n3. Gemini Analysis:")
    print("-" * 40)
    gemini = results['gemini_analysis']
    print(f"Verdict: {gemini['verdict']}")
    print(f"Confidence: {gemini['confidence']}%")
    print("\nExplanation:")
    print(gemini['explanation'])
    
    print("\nRed Flags:")
    for flag in gemini['red_flags']:
        print(f"  - {flag}")
    
    print("\nTrust Indicators:")
    for indicator in gemini['trust_indicators']:
        print(f"  - {indicator}")
    
    print("\nRecommendations:")
    for rec in gemini['recommendations']:
        print(f"  - {rec}")

def run_analysis(audio_file_path):
    """
    Run the audio analysis and print results.
    
    Args:
        audio_file_path (str): Path to the audio file to analyze.
    """
    try:
        results = analyze_audio_file(audio_file_path)
        if results:
            print_analysis_results(results)
        else:
            print("Analysis returned no results.")
    except Exception as e:
        print(f"Error: {str(e)}")

# Example usage
if __name__ == "__main__":
    audio_path = "../Audio/fraud_3 1.wav"
    run_analysis(audio_path)
