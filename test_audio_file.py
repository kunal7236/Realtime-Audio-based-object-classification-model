#!/usr/bin/env python3
"""
Test script to analyze a specific audio file and see what the model predicts
"""

import librosa
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import warnings
from pydub import AudioSegment
import os
import tempfile

def load_audio_with_fallback(file_path, sr=22050):
    """Load audio with fallback for different formats"""
    try:
        # Try librosa first
        return librosa.load(file_path, sr=sr)
    except Exception as e:
        print(f"   Librosa failed: {e}")
        print("   Trying pydub conversion...")
        
        try:
            # Convert using pydub
            audio = AudioSegment.from_file(file_path)
            # Convert to mono and resample
            audio = audio.set_channels(1).set_frame_rate(sr)
            
            # Convert to numpy array
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            samples = samples / (2**15)  # Normalize to [-1, 1]
            
            return samples, sr
        except Exception as e2:
            raise Exception(f"Both librosa and pydub failed: {e}, {e2}")

warnings.filterwarnings('ignore')

def extract_features(audio_data, sample_rate):
    """Extract the same 196 features used in training"""
    features = []
    
    # Basic audio statistics
    features.extend([
        float(np.mean(audio_data)),
        float(np.std(audio_data)),
        float(np.max(audio_data)),
        float(np.min(audio_data)),
        float(np.median(audio_data))
    ])
    
    # RMS Energy
    rms = librosa.feature.rms(y=audio_data)[0]
    features.extend([
        float(np.mean(rms)),
        float(np.std(rms)),
        float(np.max(rms)),
        float(np.min(rms))
    ])
    
    # MFCCs (13 coefficients)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    for i in range(13):
        features.extend([
            float(np.mean(mfccs[i])),
            float(np.std(mfccs[i])),
            float(np.max(mfccs[i])),
            float(np.min(mfccs[i]))
        ])
    
    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)
    features.extend([
        float(np.mean(spectral_centroids)),
        float(np.std(spectral_centroids)),
        float(np.max(spectral_centroids)),
        float(np.min(spectral_centroids))
    ])
    
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)
    features.extend([
        float(np.mean(spectral_rolloff)),
        float(np.std(spectral_rolloff)),
        float(np.max(spectral_rolloff)),
        float(np.min(spectral_rolloff))
    ])
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio_data)
    features.extend([
        float(np.mean(zcr)),
        float(np.std(zcr)),
        float(np.max(zcr)),
        float(np.min(zcr))
    ])
    
    # Spectral contrast (7 bands)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate)
    for i in range(7):
        features.extend([
            float(np.mean(spectral_contrast[i])),
            float(np.std(spectral_contrast[i])),
            float(np.max(spectral_contrast[i])),
            float(np.min(spectral_contrast[i]))
        ])
    
    # Chroma features (12 pitch classes)
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
    for i in range(12):
        features.extend([
            float(np.mean(chroma[i])),
            float(np.std(chroma[i])),
            float(np.max(chroma[i])),
            float(np.min(chroma[i]))
        ])
    
    # Tempo
    try:
        tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
        features.append(float(tempo))
    except:
        features.append(0.0)
    
    # Additional frequency domain features
    fft = np.fft.fft(audio_data)
    magnitude = np.abs(fft)
    freqs = np.fft.fftfreq(len(fft), 1/sample_rate)
    
    # Peak frequency
    peak_freq_idx = np.argmax(magnitude[:len(magnitude)//2])
    features.append(float(freqs[peak_freq_idx]))
    
    # Spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate)
    features.extend([
        float(np.mean(spectral_bandwidth)),
        float(np.std(spectral_bandwidth)),
        float(np.max(spectral_bandwidth)),
        float(np.min(spectral_bandwidth))
    ])
    
    # Pad or truncate to ensure exactly 196 features
    if len(features) < 196:
        features.extend([0.0] * (196 - len(features)))
    elif len(features) > 196:
        features = features[:196]
    
    return np.array(features, dtype=np.float32)

def test_audio_file(file_path):
    """Test the audio file with both models"""
    
    print(f"🎵 Testing audio file: {file_path}")
    print("=" * 60)
    
    # Load audio file
    try:
        audio_data, sample_rate = load_audio_with_fallback(file_path, sr=22050)
        print(f"✅ Audio loaded successfully")
        print(f"   Duration: {len(audio_data) / sample_rate:.2f} seconds")
        print(f"   Sample rate: {sample_rate} Hz")
        print(f"   Audio shape: {audio_data.shape}")
        print()
    except Exception as e:
        print(f"❌ Error loading audio: {e}")
        return
    
    # Take first 3 seconds if longer
    if len(audio_data) > 3 * sample_rate:
        audio_data = audio_data[:3 * sample_rate]
        print("📏 Using first 3 seconds of audio")
    
    # Extract features
    try:
        features = extract_features(audio_data, sample_rate)
        print(f"✅ Features extracted: {len(features)} features")
        print()
    except Exception as e:
        print(f"❌ Error extracting features: {e}")
        return
    
    # Load models and preprocessors
    try:
        # Load Random Forest (try balanced first, fallback to original)
        try:
            rf_model = joblib.load('models/balanced_random_forest.joblib')
            print("✅ Balanced Random Forest model loaded")
        except:
            rf_model = joblib.load('models/optimized_random_forest.joblib')
            print("✅ Original Random Forest model loaded (fallback)")
        
        # Load Neural Network (try balanced first, fallback to original)
        try:
            nn_model = load_model('models/balanced_neural_network.h5')
            print("✅ Balanced Neural Network model loaded")
        except:
            nn_model = load_model('models/optimized_neural_network.h5')
            print("✅ Original Neural Network model loaded (fallback)")
        
        # Load preprocessors
        scaler = joblib.load('data/features/feature_scaler.joblib')
        label_encoder = joblib.load('data/features/label_encoder.joblib')
        print("✅ Preprocessors loaded")
        print()
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return
    
    # Prepare features
    features = features.reshape(1, -1)
    features_scaled = scaler.transform(features)
    
    # Test Random Forest
    print("🌲 RANDOM FOREST PREDICTIONS:")
    print("-" * 40)
    try:
        rf_probs = rf_model.predict_proba(features_scaled)[0]
        rf_pred = rf_model.predict(features_scaled)[0]
        
        # Get class probabilities
        classes = label_encoder.classes_
        class_probs = list(zip(classes, rf_probs))
        class_probs.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Top prediction: {label_encoder.inverse_transform([rf_pred])[0]} (confidence: {np.max(rf_probs):.3f})")
        print("All predictions:")
        for class_name, prob in class_probs:
            print(f"   {class_name:20s}: {prob:.3f}")
        print()
        
    except Exception as e:
        print(f"❌ Random Forest error: {e}")
    
    # Test Neural Network
    print("🧠 NEURAL NETWORK PREDICTIONS:")
    print("-" * 40)
    try:
        nn_probs = nn_model.predict(features_scaled, verbose=0)[0]
        nn_pred = np.argmax(nn_probs)
        
        # Get class probabilities
        class_probs = list(zip(classes, nn_probs))
        class_probs.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Top prediction: {classes[nn_pred]} (confidence: {np.max(nn_probs):.3f})")
        print("All predictions:")
        for class_name, prob in class_probs:
            print(f"   {class_name:20s}: {prob:.3f}")
        print()
        
    except Exception as e:
        print(f"❌ Neural Network error: {e}")
    
    # Audio analysis
    print("📊 AUDIO ANALYSIS:")
    print("-" * 40)
    rms_energy = np.sqrt(np.mean(audio_data**2))
    max_amplitude = np.max(np.abs(audio_data))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate))
    
    print(f"RMS Energy: {rms_energy:.4f}")
    print(f"Max Amplitude: {max_amplitude:.4f}")
    print(f"Spectral Centroid: {spectral_centroid:.1f} Hz")
    
    # Check if it would be classified as silence
    silence_thresholds = {
        'rms': 0.005,
        'max': 0.02,
        'spectral_centroid_min': 1000,
        'spectral_centroid_max': 3000
    }
    
    is_silence = (rms_energy < silence_thresholds['rms'] and 
                 max_amplitude < silence_thresholds['max'])
    
    print(f"Would be classified as silence: {is_silence}")

if __name__ == "__main__":
    file_path = r"C:\Users\kunal\OneDrive\Desktop\youtube-download\abc.wav"
    test_audio_file(file_path)
