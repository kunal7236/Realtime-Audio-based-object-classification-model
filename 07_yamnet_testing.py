"""
YAMNet Model Testing Pipeline
Replaces traditional model testing with YAMNet evaluation on test split
"""

import numpy as np
import pandas as pd
import librosa
import tensorflow_hub as hub
import joblib
from tensorflow.keras.models import load_model
import warnings
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

warnings.filterwarnings('ignore')

class YAMNetTester:
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
    
    def test_models(self):
        """Test models on test split"""
        print(f"\n🎯 Testing models on TEST split")
        print("=" * 60)
        
        # Load test data
        df = pd.read_csv('data/data_splits.csv')
        test_data = df[df['split'] == 'test'].copy()
        
        if len(test_data) == 0:
            print(f"❌ No test data found")
            return None
        
        print(f"📊 Total test samples: {len(test_data)}")
        
        # Initialize results
        y_true = []
        y_pred_rf = []
        y_pred_nn = []
        rf_confidences = []
        nn_confidences = []
        failed_files = []
        
        # Process each audio file
        print(f"🔧 Processing {len(test_data)} test files...")
        
        for idx, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Testing"):
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
    
    def print_test_results(self, results):
        """Print detailed test results"""
        print(f"\n📈 FINAL TEST RESULTS")
        print("=" * 60)
        
        # Overall accuracies
        rf_accuracy = accuracy_score(results['y_true'], results['y_pred_rf'])
        nn_accuracy = accuracy_score(results['y_true'], results['y_pred_nn'])
        
        print(f"🌲 Random Forest Test Accuracy: {rf_accuracy:.3f} ({rf_accuracy*100:.1f}%)")
        print(f"🧠 Neural Network Test Accuracy: {nn_accuracy:.3f} ({nn_accuracy*100:.1f}%)")
        
        # Average confidences
        rf_avg_conf = np.mean(results['rf_confidences'])
        nn_avg_conf = np.mean(results['nn_confidences'])
        
        print(f"🌲 RF Average Confidence: {rf_avg_conf:.3f}")
        print(f"🧠 NN Average Confidence: {nn_avg_conf:.3f}")
        
        # Per-class performance
        print(f"\n📊 PER-CLASS TEST PERFORMANCE:")
        print("-" * 50)
        print(f"{'Class':15} {'Samples':>8} {'RF Acc':>8} {'NN Acc':>8} {'Best':>8}")
        print("-" * 50)
        
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
            
            # Best model for this class
            best = "RF" if rf_acc > nn_acc else "NN" if nn_acc > rf_acc else "Tie"
            
            print(f"{class_name:15} {total_samples:8d} {rf_acc:8.3f} {nn_acc:8.3f} {best:>8}")
        
        # Classification reports
        print(f"\n🌲 RANDOM FOREST FINAL REPORT:")
        print(classification_report(results['y_true'], results['y_pred_rf'], 
                                  target_names=self.class_names, digits=3))
        
        print(f"\n🧠 NEURAL NETWORK FINAL REPORT:")
        print(classification_report(results['y_true'], results['y_pred_nn'], 
                                  target_names=self.class_names, digits=3))
        
        return rf_accuracy, nn_accuracy
    
    def save_confusion_matrices(self, results):
        """Save confusion matrices as images"""
        print(f"\n💾 Saving confusion matrices...")
        
        os.makedirs('results/testing', exist_ok=True)
        
        # Random Forest confusion matrix
        plt.figure(figsize=(10, 8))
        cm_rf = confusion_matrix(results['y_true'], results['y_pred_rf'], labels=self.class_names)
        sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Random Forest - Final Test Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('results/testing/final_confusion_matrix_rf.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Neural Network confusion matrix
        plt.figure(figsize=(10, 8))
        cm_nn = confusion_matrix(results['y_true'], results['y_pred_nn'], labels=self.class_names)
        sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Greens', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Neural Network - Final Test Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('results/testing/final_confusion_matrix_nn.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Confusion matrices saved to results/testing/")
    
    def save_test_report(self, results):
        """Save comprehensive test report"""
        print(f"\n💾 Saving final test report...")
        
        os.makedirs('results/testing', exist_ok=True)
        
        # Create comprehensive report
        report = []
        report.append("YAMNet Models - Final Test Report")
        report.append("=" * 60)
        report.append(f"Test samples processed: {len(results['y_true'])}")
        report.append(f"Failed files: {len(results['failed_files'])}")
        report.append("")
        
        # Overall accuracies
        rf_accuracy = accuracy_score(results['y_true'], results['y_pred_rf'])
        nn_accuracy = accuracy_score(results['y_true'], results['y_pred_nn'])
        
        report.append("FINAL MODEL PERFORMANCE:")
        report.append(f"Random Forest Test Accuracy: {rf_accuracy:.3f} ({rf_accuracy*100:.1f}%)")
        report.append(f"Neural Network Test Accuracy: {nn_accuracy:.3f} ({nn_accuracy*100:.1f}%)")
        report.append("")
        
        # Model recommendation
        if nn_accuracy > rf_accuracy:
            better_model = "Neural Network"
            improvement = (nn_accuracy - rf_accuracy) * 100
            report.append(f"RECOMMENDATION: Use Neural Network (+{improvement:.1f}% better)")
        elif rf_accuracy > nn_accuracy:
            better_model = "Random Forest"
            improvement = (rf_accuracy - nn_accuracy) * 100
            report.append(f"RECOMMENDATION: Use Random Forest (+{improvement:.1f}% better)")
        else:
            better_model = "Both models equivalent"
            report.append(f"RECOMMENDATION: Both models perform equally well")
        
        report.append("")
        
        # Per-class breakdown
        report.append("PER-CLASS PERFORMANCE:")
        report.append("-" * 50)
        report.append(f"{'Class':15} {'Samples':>8} {'RF Acc':>8} {'NN Acc':>8} {'Best':>8}")
        report.append("-" * 50)
        
        for class_name in sorted(self.class_names):
            class_indices = [i for i, label in enumerate(results['y_true']) if label == class_name]
            
            if not class_indices:
                continue
                
            total_samples = len(class_indices)
            rf_correct = sum(1 for i in class_indices if results['y_pred_rf'][i] == class_name)
            nn_correct = sum(1 for i in class_indices if results['y_pred_nn'][i] == class_name)
            
            rf_acc = rf_correct / total_samples
            nn_acc = nn_correct / total_samples
            best = "RF" if rf_acc > nn_acc else "NN" if nn_acc > rf_acc else "Tie"
            
            report.append(f"{class_name:15} {total_samples:8d} {rf_acc:8.3f} {nn_acc:8.3f} {best:>8}")
        
        # Save to file
        with open('results/testing/final_test_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        print(f"✅ Final test report saved to results/testing/final_test_report.txt")
        
        return better_model, max(rf_accuracy, nn_accuracy)

def main():
    """Main testing pipeline"""
    print("🎯 YAMNet Model Final Testing Pipeline")
    print("=" * 60)
    
    # Initialize tester
    tester = YAMNetTester()
    tester.load_models()
    
    # Run final tests
    results = tester.test_models()
    
    if results is not None:
        rf_acc, nn_acc = tester.print_test_results(results)
        tester.save_confusion_matrices(results)
        better_model, best_acc = tester.save_test_report(results)
        
        # Final summary
        print(f"\n🎉 FINAL TESTING COMPLETE!")
        print("=" * 60)
        print(f"🌲 Random Forest Final Accuracy: {rf_acc:.3f} ({rf_acc*100:.1f}%)")
        print(f"🧠 Neural Network Final Accuracy: {nn_acc:.3f} ({nn_acc*100:.1f}%)")
        print(f"🏆 Best Model: {better_model}")
        print(f"📊 Best Accuracy: {best_acc:.3f} ({best_acc*100:.1f}%)")
        print(f"📁 Results saved to results/testing/")
        print(f"🚀 Models ready for deployment!")
    else:
        print("❌ Testing failed - no test data found")

if __name__ == "__main__":
    main()
