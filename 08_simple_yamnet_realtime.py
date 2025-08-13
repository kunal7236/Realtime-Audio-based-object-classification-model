
"""
Simple YAMNet Enhanced Real-Time Audio Classifier
Using our trained YAMNet enhanced models
"""

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pyaudio
import time
import threading
import queue
import joblib
from collections import deque
import warnings

warnings.filterwarnings('ignore')

class YAMNetRealTimeClassifier:
    def __init__(self):
        print("🎯 YAMNet Enhanced Real-Time Audio Classifier")
        print("For Assisting Visually Impaired People")
        print()
        
        # Audio settings
        self.sample_rate = 16000
        self.chunk_duration = 3.0  # 3 seconds
        self.chunk_samples = int(self.sample_rate * self.chunk_duration)
        
        # Detection settings
        self.confidence_threshold = 0.6  # 60% confidence minimum
        self.alert_threshold = 0.8       # 80% for safety alerts
        
        # Audio processing
        self.audio_queue = queue.Queue()
        self.audio_buffer = deque(maxlen=self.chunk_samples)
        self.running = False
        
        # Safety sounds that need immediate attention
        self.safety_sounds = ['car_horn', 'siren', 'dog_bark']
        
        # Load our trained models
        self.load_yamnet_models()
    
    def load_yamnet_models(self):
        """Load our YAMNet enhanced models"""
        print("🔄 Loading YAMNet Enhanced Models...")
        
        try:
            # Load YAMNet from TensorFlow Hub
            print("📥 Loading YAMNet...")
            self.yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
            print("✅ YAMNet loaded successfully")
            
            # Load our enhanced classifiers
            print("📦 Loading our enhanced classifiers...")
            self.random_forest = joblib.load('models/enhanced_yamnet_random_forest.joblib')
            self.neural_network = tf.keras.models.load_model('models/enhanced_yamnet_neural_network.keras')
            self.label_encoder = joblib.load('models/enhanced_yamnet_label_encoder.joblib')
            print("✅ Enhanced classifiers loaded successfully")
            
            # Show what we can detect
            classes = self.label_encoder.classes_
            print(f"📊 Can detect {len(classes)} sound types:")
            for i, sound_class in enumerate(classes):
                icon = self.get_sound_icon(sound_class)
                print(f"   {i+1}. {icon} {sound_class.replace('_', ' ').title()}")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
    
    def get_sound_icon(self, sound_class):
        """Get emoji icon for sound class"""
        icons = {
            'car_horn': '🚗',
            'dog_bark': '🐕',
            'siren': '🚨',
            'children_playing': '👶',
            'air_conditioner': '❄️',
            'drilling': '🔨',
            'engine_idling': '🚛',
            'jackhammer': '⚡',
            'street_music': '🎵'
        }
        return icons.get(sound_class, '🔊')
    
    def extract_yamnet_features(self, audio_chunk):
        """Extract YAMNet embeddings from audio chunk"""
        try:
            # Ensure audio is float32 and properly normalized
            if audio_chunk.dtype != np.float32:
                audio_chunk = audio_chunk.astype(np.float32)
            
            # Normalize audio to [-1, 1] range
            max_val = np.max(np.abs(audio_chunk))
            if max_val > 1.0:
                audio_chunk = audio_chunk / max_val
            
            # Get YAMNet embeddings
            scores, embeddings, spectrogram = self.yamnet_model(audio_chunk)
            
            # Use mean of all embeddings as features
            features = np.mean(embeddings.numpy(), axis=0)
            
            return features.reshape(1, -1)
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None
    
    def classify_audio(self, features):
        """Classify audio using both our enhanced models"""
        try:
            # Random Forest prediction
            rf_probabilities = self.random_forest.predict_proba(features)[0]
            rf_prediction = np.argmax(rf_probabilities)
            rf_confidence = rf_probabilities[rf_prediction]
            rf_class = self.label_encoder.classes_[rf_prediction]
            
            # Neural Network prediction
            nn_probabilities = self.neural_network.predict(features, verbose=0)[0]
            nn_prediction = np.argmax(nn_probabilities)
            nn_confidence = nn_probabilities[nn_prediction]
            nn_class = self.label_encoder.classes_[nn_prediction]
            
            # Determine final prediction
            if rf_class == nn_class:
                # Both models agree
                final_class = rf_class
                final_confidence = (rf_confidence + nn_confidence) / 2
                agreement = "Both Models Agree"
            else:
                # Models disagree - use higher confidence
                if rf_confidence > nn_confidence:
                    final_class = rf_class
                    final_confidence = rf_confidence
                    agreement = "Random Forest"
                else:
                    final_class = nn_class
                    final_confidence = nn_confidence
                    agreement = "Neural Network"
            
            return final_class, final_confidence, agreement
            
        except Exception as e:
            print(f"Classification error: {e}")
            return None, 0, "Error"
    
    def format_detection_message(self, sound_class, confidence, model_info):
        """Format the detection message"""
        icon = self.get_sound_icon(sound_class)
        sound_name = sound_class.replace('_', ' ').title()
        
        # Determine alert level based on confidence
        if confidence >= self.alert_threshold:
            if sound_class in self.safety_sounds:
                alert = "🔴 SAFETY ALERT"
            else:
                alert = "🟢 DETECTED"
        else:
            alert = "🟡 DETECTED"
        
        return f"{alert} {icon} {sound_name} - {confidence:.1%} confidence ({model_info})"
    
    def audio_input_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio input stream"""
        try:
            # Convert audio data to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Add to processing queue
            self.audio_queue.put(audio_data)
            
        except Exception as e:
            print(f"Audio input error: {e}")
        
        return (None, pyaudio.paContinue)
    
    def audio_processing_thread(self):
        """Process audio in separate thread"""
        print("🎵 Audio processing thread started...")
        
        while self.running:
            try:
                # Get audio data from queue
                try:
                    audio_data = self.audio_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Add to buffer
                self.audio_buffer.extend(audio_data)
                
                # Process when we have enough data
                if len(self.audio_buffer) >= self.chunk_samples:
                    # Get audio chunk
                    chunk = np.array(list(self.audio_buffer))
                    
                    # Extract YAMNet features
                    features = self.extract_yamnet_features(chunk)
                    
                    if features is not None:
                        # Classify the audio
                        sound_class, confidence, model_info = self.classify_audio(features)
                        
                        # Only report if confidence is above threshold
                        if sound_class and confidence >= self.confidence_threshold:
                            # Format message
                            message = self.format_detection_message(sound_class, confidence, model_info)
                            timestamp = time.strftime('%H:%M:%S')
                            
                            # Display detection
                            print(f"[{timestamp}] {message}")
                            
                            # Show agreement indicator
                            if model_info == "Both Models Agree":
                                print("✅ Model Agreement")
                            
                            # Safety alert for critical sounds
                            if sound_class in self.safety_sounds and confidence >= self.alert_threshold:
                                print("🚨 IMMEDIATE ATTENTION REQUIRED! 🚨")
                                print()
                    
                    # Clear some buffer for overlap
                    clear_amount = len(self.audio_buffer) // 2  # Clear half buffer
                    for _ in range(clear_amount):
                        if self.audio_buffer:
                            self.audio_buffer.popleft()
                
            except Exception as e:
                print(f"Processing thread error: {e}")
                continue
    
    def start_real_time_detection(self):
        """Start real-time audio detection"""
        print()
        print("🚀 Starting Real-Time Audio Detection")
        print("=" * 50)
        print(f"📊 Settings:")
        print(f"   Sample Rate: {self.sample_rate} Hz")
        print(f"   Chunk Duration: {self.chunk_duration} seconds")
        print(f"   Confidence Threshold: {self.confidence_threshold:.0%}")
        print(f"   Alert Threshold: {self.alert_threshold:.0%}")
        print("=" * 50)
        print("🔊 Listening for sounds... (Press Ctrl+C to stop)")
        print()
        
        # Initialize PyAudio
        audio = pyaudio.PyAudio()
        
        try:
            # Open microphone stream
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=1024,
                stream_callback=self.audio_input_callback
            )
            
            # Start processing thread
            self.running = True
            processing_thread = threading.Thread(target=self.audio_processing_thread)
            processing_thread.daemon = True
            processing_thread.start()
            
            # Start audio stream
            stream.start_stream()
            
            # Keep running until interrupted
            try:
                while stream.is_active() and self.running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n🛑 Stopping real-time detection...")
                self.running = False
            
            # Cleanup
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            print(f"❌ Audio stream error: {e}")
        finally:
            audio.terminate()
            print("✅ Real-time detection stopped")

def main():
    """Main function"""
    try:
        # Create and start the classifier
        classifier = YAMNetRealTimeClassifier()
        classifier.start_real_time_detection()
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
