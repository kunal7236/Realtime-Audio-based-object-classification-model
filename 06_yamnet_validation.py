"""
YAMNet Model Validation Pipeline
Replaces traditional model validation with YAMNet evaluation
"""

import numpy as np
import pandas as pd
import librosa
import tensorflow_hub as hub
import joblib
from tensorflow.keras.models import load_model
import warnings
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import os
from tqdm import tqdm

warnings.filterwarnings('ignore')

class YAMNetValidator:
    def __init__(self):
        self.yamnet_model = None
        self.rf_model = None
        self.nn_model = None
        self.label_encoder = None
        self.class_names = None
        
    def load_models(self):
        """Load all trained models"""
        print("📥 Loading trained models...")
        
        try:
            # Load YAMNet
            self.yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
            
            # Load trained models
            self.rf_model = joblib.load('models/yamnet_random_forest.joblib')
            self.nn_model = load_model('models/yamnet_neural_network.keras')
            self.label_encoder = joblib.load('models/yamnet_label_encoder.joblib')
            
            self.class_names = self.label_encoder.classes_
            print(f"✅ All models loaded. Classes: {list(self.class_names)}")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
        
    def extract_yamnet_features(self, audio_file):
        """Extract YAMNet embeddings from audio file"""
        try:
            # Load audio at 16kHz for YAMNet
            audio, sr = librosa.load(audio_file, sr=16000, duration=3.0)
            
            # Ensure exactly 3 seconds
            target_length = 16000 * 3
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)))
            else:
                audio = audio[:target_length]
            
            # Extract YAMNet embeddings
            embeddings, _, _ = self.yamnet_model(audio.astype(np.float32))
            # Average across time
            avg_embedding = np.mean(embeddings.numpy(), axis=0)
            
            return avg_embedding
            
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            return np.zeros(521)  # YAMNet embedding size
    
    def validate_on_split(self, split_name='validation'):
        """Validate models on specified data split"""
        print(f"\n🎯 Validating on {split_name.upper()} split")
        print("=" * 60)
        
        # Load data splits
        df = pd.read_csv('data/data_splits.csv')
        split_data = df[df['split'] == split_name].copy()
        
        if len(split_data) == 0:
            print(f"❌ No data found for {split_name} split")
            return None
        
        print(f"📊 Total {split_name} samples: {len(split_data)}")
        
        # Initialize results
        y_true = []
        y_pred_rf = []
        y_pred_nn = []
        rf_confidences = []
        nn_confidences = []
        failed_files = []
        
        # Process each audio file
        print(f"🔧 Processing {len(split_data)} audio files...")
        
        for idx, row in tqdm(split_data.iterrows(), total=len(split_data), desc="Processing"):
            audio_path = os.path.join('data/UrbanSound8K/audio', f"fold{row['fold']}", row['slice_file_name'])
            
            if not os.path.exists(audio_path):
                failed_files.append(row['slice_file_name'])
                continue
            
            # Extract features
            features = self.extract_yamnet_features(audio_path)
            if np.all(features == 0):
                failed_files.append(row['slice_file_name'])
                continue
            
            features = features.reshape(1, -1)
            
            # Get true label
            true_label = row['class']
            y_true.append(true_label)
            
            # Random Forest prediction
            rf_pred_proba = self.rf_model.predict_proba(features)[0]
            rf_pred_idx = np.argmax(rf_pred_proba)
            rf_pred_label = self.label_encoder.inverse_transform([rf_pred_idx])[0]
            rf_confidence = rf_pred_proba[rf_pred_idx]
            
            y_pred_rf.append(rf_pred_label)
            rf_confidences.append(rf_confidence)
            
            # Neural Network prediction
            nn_pred_proba = self.nn_model.predict(features, verbose=0)[0]
            nn_pred_idx = np.argmax(nn_pred_proba)
            nn_pred_label = self.label_encoder.inverse_transform([nn_pred_idx])[0]
            nn_confidence = nn_pred_proba[nn_pred_idx]
            
            y_pred_nn.append(nn_pred_label)
            nn_confidences.append(nn_confidence)
        
        print(f"✅ Processed {len(y_true)} files successfully")
        if failed_files:
            print(f"⚠️  Failed to process {len(failed_files)} files")
        
        return {
            'y_true': y_true,
            'y_pred_rf': y_pred_rf,
            'y_pred_nn': y_pred_nn,
            'rf_confidences': rf_confidences,
            'nn_confidences': nn_confidences,
            'failed_files': failed_files
        }
    
    def print_validation_results(self, results, split_name):
        """Print detailed validation results"""
        print(f"\n📈 VALIDATION RESULTS FOR {split_name.upper()} SPLIT")
        print("=" * 60)
        
        # Overall accuracies
        rf_accuracy = accuracy_score(results['y_true'], results['y_pred_rf'])
        nn_accuracy = accuracy_score(results['y_true'], results['y_pred_nn'])
        
        print(f"🌲 Random Forest Accuracy: {rf_accuracy:.3f} ({rf_accuracy*100:.1f}%)")
        print(f"🧠 Neural Network Accuracy: {nn_accuracy:.3f} ({nn_accuracy*100:.1f}%)")
        
        # Average confidences
        rf_avg_conf = np.mean(results['rf_confidences'])
        nn_avg_conf = np.mean(results['nn_confidences'])
        
        print(f"🌲 RF Average Confidence: {rf_avg_conf:.3f}")
        print(f"🧠 NN Average Confidence: {nn_avg_conf:.3f}")
        
        # Per-class performance
        print(f"\n📊 PER-CLASS PERFORMANCE:")
        print("-" * 40)
        
        for class_name in sorted(self.class_names):
            # Count samples for this class
            class_indices = [i for i, label in enumerate(results['y_true']) if label == class_name]
            
            if not class_indices:
                continue
                
            total_samples = len(class_indices)
            
            # RF performance
            rf_correct = sum(1 for i in class_indices if results['y_pred_rf'][i] == class_name)
            rf_acc = rf_correct / total_samples
            
            # NN performance  
            nn_correct = sum(1 for i in class_indices if results['y_pred_nn'][i] == class_name)
            nn_acc = nn_correct / total_samples
            
            print(f"{class_name:15} ({total_samples:3d} samples): RF {rf_acc:.3f} | NN {nn_acc:.3f}")
        
        # Classification reports
        print(f"\n🌲 RANDOM FOREST DETAILED REPORT:")
        print(classification_report(results['y_true'], results['y_pred_rf'], 
                                  target_names=self.class_names, digits=3))
        
        print(f"\n🧠 NEURAL NETWORK DETAILED REPORT:")
        print(classification_report(results['y_true'], results['y_pred_nn'], 
                                  target_names=self.class_names, digits=3))
        
        return rf_accuracy, nn_accuracy
    
    def save_validation_report(self, results, split_name):
        """Save validation results to file"""
        print(f"\n💾 Saving validation report...")
        
        os.makedirs('results/validation', exist_ok=True)
        
        # Create report
        report = []
        report.append(f"YAMNet Model Validation Report - {split_name.upper()} Split")
        report.append("=" * 60)
        report.append(f"Total samples processed: {len(results['y_true'])}")
        report.append(f"Failed files: {len(results['failed_files'])}")
        report.append("")
        
        # Overall accuracies
        rf_accuracy = accuracy_score(results['y_true'], results['y_pred_rf'])
        nn_accuracy = accuracy_score(results['y_true'], results['y_pred_nn'])
        
        report.append(f"Random Forest Accuracy: {rf_accuracy:.3f} ({rf_accuracy*100:.1f}%)")
        report.append(f"Neural Network Accuracy: {nn_accuracy:.3f} ({nn_accuracy*100:.1f}%)")
        report.append("")
        
        # Per-class breakdown
        report.append("Per-Class Performance:")
        report.append("-" * 40)
        
        for class_name in sorted(self.class_names):
            class_indices = [i for i, label in enumerate(results['y_true']) if label == class_name]
            
            if not class_indices:
                continue
                
            total_samples = len(class_indices)
            rf_correct = sum(1 for i in class_indices if results['y_pred_rf'][i] == class_name)
            nn_correct = sum(1 for i in class_indices if results['y_pred_nn'][i] == class_name)
            
            rf_acc = rf_correct / total_samples
            nn_acc = nn_correct / total_samples
            
            report.append(f"{class_name:15} ({total_samples:3d} samples): RF {rf_acc:.3f} | NN {nn_acc:.3f}")
        
        # Save to file
        with open(f'results/validation/validation_report_{split_name}.txt', 'w') as f:
            f.write('\n'.join(report))
        
        print(f"✅ Validation report saved to results/validation/validation_report_{split_name}.txt")

def main():
    """Main validation pipeline"""
    print("🎯 YAMNet Model Validation Pipeline")
    print("=" * 60)
    
    # Initialize validator
    validator = YAMNetValidator()
    validator.load_models()
    
    # Validate on validation split
    results = validator.validate_on_split('validation')
    
    if results is not None:
        rf_acc, nn_acc = validator.print_validation_results(results, 'validation')
        validator.save_validation_report(results, 'validation')
        
        # Summary
        print(f"\n🎉 VALIDATION COMPLETE!")
        print("-" * 40)
        print(f"🌲 Random Forest Accuracy: {rf_acc:.3f} ({rf_acc*100:.1f}%)")
        print(f"🧠 Neural Network Accuracy: {nn_acc:.3f} ({nn_acc*100:.1f}%)")
        print(f"📁 Report saved to results/validation/")
        
        # Model recommendation
        if nn_acc > rf_acc:
            print(f"💡 Recommendation: Neural Network performs better (+{(nn_acc-rf_acc)*100:.1f}%)")
        elif rf_acc > nn_acc:
            print(f"💡 Recommendation: Random Forest performs better (+{(rf_acc-nn_acc)*100:.1f}%)")
        else:
            print(f"💡 Both models perform equally well")
    else:
        print("❌ Validation failed - no validation data found")

if __name__ == "__main__":
    main()
