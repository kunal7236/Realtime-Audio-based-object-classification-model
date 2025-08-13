"""
Enhanced YAMNet-based Audio Classification Training
With Aggressive Class Balancing for Rare Classes
"""

import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class EnhancedYAMNetTrainer:
    def __init__(self):
        self.yamnet_model = None
        self.rf_model = None
        self.nn_model = None
        self.label_encoder = LabelEncoder()
        self.class_names = None
        
    def load_yamnet(self):
        """Load pre-trained YAMNet model"""
        print("📥 Loading YAMNet model...")
        self.yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
        print("✅ YAMNet loaded")
    
    def extract_yamnet_features(self, audio_path):
        """Extract YAMNet embeddings from audio file"""
        try:
            # Load audio with YAMNet's expected sample rate (16kHz)
            y, sr = librosa.load(audio_path, sr=16000, duration=3.0)
            
            # Ensure we have 3 seconds of audio (pad if necessary)
            target_length = 16000 * 3  # 3 seconds at 16kHz
            if len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)), mode='constant')
            elif len(y) > target_length:
                y = y[:target_length]
            
            # Convert to TensorFlow tensor
            waveform = tf.convert_to_tensor(y, dtype=tf.float32)
            
            # Get YAMNet embeddings
            _, embeddings, _ = self.yamnet_model(waveform)
            
            # Average embeddings across time
            feature_vector = tf.reduce_mean(embeddings, axis=0).numpy()
            
            return feature_vector
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return np.zeros(521)  # YAMNet embedding size is 1024, but we'll check this
    
    def prepare_data(self, split='train'):
        """Prepare training data with YAMNet features"""
        print(f"\n📊 Preparing {split} data...")
        
        # Load data splits
        splits_df = pd.read_csv('data/data_splits.csv')
        split_data = splits_df[splits_df['split'] == split]
        
        # Show class distribution for this split
        if split == 'train':
            print(f"📊 Class distribution in {split}:")
            class_counts = split_data['class'].value_counts().sort_index()
            total = len(split_data)
            for class_name, count in class_counts.items():
                percentage = (count / total) * 100
                print(f"   {class_name:15}: {count:3d} ({percentage:4.1f}%)")
        else:
            print(f"📊 Total {split} samples: {len(split_data)}")
        
        # Extract features
        features = []
        labels = []
        failed_count = 0
        
        print(f"🎵 Extracting YAMNet features...")
        for idx, row in tqdm(split_data.iterrows(), total=len(split_data), desc="Processing"):
            audio_path = os.path.join('data/UrbanSound8K/audio', f"fold{row['fold']}", row['slice_file_name'])
            
            if not os.path.exists(audio_path):
                failed_count += 1
                continue
            
            # Extract features
            feature_vector = self.extract_yamnet_features(audio_path)
            if np.all(feature_vector == 0):
                failed_count += 1
                continue
            
            features.append(feature_vector)
            labels.append(row['class'])
        
        if failed_count > 0:
            print(f"⚠️  Failed to process {failed_count} files")
        
        print(f"✅ Extracted features: {len(features)} samples, {len(features[0]) if features else 0} features")
        
        return np.array(features), np.array(labels)
    
    def calculate_enhanced_class_weights(self, y_train):
        """Calculate enhanced class weights with extra penalty for rare classes"""
        print("\n⚖️  Calculating enhanced class weights...")
        
        # Get class counts
        unique_classes, counts = np.unique(y_train, return_counts=True)
        total_samples = len(y_train)
        
        # Calculate enhanced weights
        enhanced_weights = {}
        for class_name, count in zip(unique_classes, counts):
            percentage = (count / total_samples) * 100
            
            # Base balanced weight
            base_weight = total_samples / (len(unique_classes) * count)
            
            # Apply multiplier for rare classes
            if percentage < 5.0:  # Very rare classes (like car_horn at 4.8%)
                multiplier = 2.0
                print(f"   🔴 {class_name:15}: {count:3d} ({percentage:4.1f}%) -> 2.0x boost")
            elif percentage < 8.0:  # Moderately rare classes
                multiplier = 1.5
                print(f"   🟡 {class_name:15}: {count:3d} ({percentage:4.1f}%) -> 1.5x boost")
            else:  # Common classes
                multiplier = 1.0
                print(f"   🟢 {class_name:15}: {count:3d} ({percentage:4.1f}%) -> normal")
            
            enhanced_weights[class_name] = base_weight * multiplier
        
        return enhanced_weights
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest classifier with enhanced class weights"""
        print("\n🌲 Training Enhanced Random Forest...")
        
        # Calculate enhanced class weights
        enhanced_weights = self.calculate_enhanced_class_weights(y_train)
        
        self.rf_model = RandomForestClassifier(
            n_estimators=300,  # Increased from 200
            max_depth=25,      # Increased from 20
            min_samples_split=3,  # Decreased from 5 for better fitting
            min_samples_leaf=1,   # Decreased from 2
            class_weight=enhanced_weights,  # Use enhanced weights
            random_state=42,
            n_jobs=-1
        )
        
        self.rf_model.fit(X_train, y_train)
        print("✅ Enhanced Random Forest trained")
        
    def train_neural_network(self, X_train, y_train, X_val=None, y_val=None):
        """Train Neural Network classifier with enhanced class weights"""
        print("\n🧠 Training Enhanced Neural Network...")
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        n_classes = len(self.label_encoder.classes_)
        self.class_names = self.label_encoder.classes_
        
        if X_val is not None and y_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)
            validation_data = (X_val, y_val_encoded)
        else:
            validation_data = None
        
        # Calculate enhanced class weights
        enhanced_weights = self.calculate_enhanced_class_weights(y_train)
        
        # Convert to numeric indices for Keras
        class_weight_dict = {}
        for i, class_name in enumerate(self.label_encoder.classes_):
            class_weight_dict[i] = enhanced_weights[class_name]
        
        print(f"🎯 Enhanced class weights for NN:")
        for i, (class_name, weight) in enumerate(enhanced_weights.items()):
            print(f"   {class_name:15}: {weight:.3f}")
        
        # Build enhanced model
        self.nn_model = Sequential([
            Dense(512, activation='relu', input_shape=(1024,)),  # YAMNet outputs 1024-dim embeddings
            BatchNormalization(),
            Dropout(0.4),  # Increased dropout
            
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(64, activation='relu'),
            Dropout(0.1),
            
            Dense(n_classes, activation='softmax')
        ])
        
        # Compile with lower learning rate for better convergence
        self.nn_model.compile(
            optimizer=Adam(learning_rate=0.0005),  # Reduced from 0.001
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Enhanced callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=15,  # Increased patience
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'models/enhanced_yamnet_neural_network.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train with enhanced settings
        history = self.nn_model.fit(
            X_train, y_train_encoded,
            validation_data=validation_data,
            epochs=100,  # Increased epochs
            batch_size=32,
            class_weight=class_weight_dict,  # Enhanced class weights
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Enhanced Neural Network trained")
        return history
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate both models"""
        print("\n📊 Evaluating Enhanced Models...")
        
        # Random Forest evaluation
        print("\n🌲 ENHANCED RANDOM FOREST RESULTS:")
        rf_pred = self.rf_model.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        print(f"Accuracy: {rf_accuracy:.4f}")
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, rf_pred, zero_division=0))
        
        # Neural Network evaluation
        print("\n🧠 ENHANCED NEURAL NETWORK RESULTS:")
        y_test_encoded = self.label_encoder.transform(y_test)
        nn_pred_proba = self.nn_model.predict(X_test, verbose=0)
        nn_pred = np.argmax(nn_pred_proba, axis=1)
        nn_pred_labels = self.label_encoder.inverse_transform(nn_pred)
        nn_accuracy = accuracy_score(y_test, nn_pred_labels)
        print(f"Accuracy: {nn_accuracy:.4f}")
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, nn_pred_labels, zero_division=0))
        
        # # Focus on car_horn performance
        # print("\n🚗 CAR HORN SPECIFIC PERFORMANCE:")
        # car_horn_indices = [i for i, label in enumerate(y_test) if label == 'car_horn']
        # if car_horn_indices:
        #     car_horn_true = [y_test[i] for i in car_horn_indices]
        #     car_horn_rf_pred = [rf_pred[i] for i in car_horn_indices]
        #     car_horn_nn_pred = [nn_pred_labels[i] for i in car_horn_indices]
            
        #     rf_car_accuracy = sum(1 for t, p in zip(car_horn_true, car_horn_rf_pred) if t == p) / len(car_horn_true)
        #     nn_car_accuracy = sum(1 for t, p in zip(car_horn_true, car_horn_nn_pred) if t == p) / len(car_horn_true)
            
        #     print(f"RF car_horn accuracy: {rf_car_accuracy:.3f} ({sum(1 for t, p in zip(car_horn_true, car_horn_rf_pred) if t == p)}/{len(car_horn_true)})")
        #     print(f"NN car_horn accuracy: {nn_car_accuracy:.3f} ({sum(1 for t, p in zip(car_horn_true, car_horn_nn_pred) if t == p)}/{len(car_horn_true)})")
        
        return rf_accuracy, nn_accuracy
    
    def save_models(self):
        """Save trained models"""
        print("\n💾 Saving enhanced models...")
        
        # Save Random Forest
        joblib.dump(self.rf_model, 'models/enhanced_yamnet_random_forest.joblib')
        print("✅ Enhanced Random Forest saved")
        
        # Save Neural Network (already saved by callback)
        print("✅ Enhanced Neural Network saved")
        
        # Save label encoder
        joblib.dump(self.label_encoder, 'models/enhanced_yamnet_label_encoder.joblib')
        print("✅ Enhanced Label encoder saved")

def main():
    print("🚀 Enhanced YAMNet Training with Aggressive Class Balancing")
    print("=" * 60)
    
    # Initialize trainer
    trainer = EnhancedYAMNetTrainer()
    
    # Load YAMNet
    trainer.load_yamnet()
    
    # Prepare data
    print("\n🔄 Preparing enhanced training data...")
    X_train, y_train = trainer.prepare_data('train')
    X_val, y_val = trainer.prepare_data('validation')
    X_test, y_test = trainer.prepare_data('test')
    
    # Train models with enhanced class balancing
    trainer.train_random_forest(X_train, y_train)
    trainer.train_neural_network(X_train, y_train, X_val, y_val)
    
    # Evaluate models
    rf_acc, nn_acc = trainer.evaluate_models(X_test, y_test)
    
    # Save models
    trainer.save_models()
    
    print(f"\n🎉 Enhanced Training Complete!")
    print(f"📈 Final Results:")
    print(f"   Enhanced Random Forest: {rf_acc:.4f}")
    print(f"   Enhanced Neural Network: {nn_acc:.4f}")
    
    if rf_acc > 0.85 or nn_acc > 0.85:
        print("🎯 Excellent accuracy achieved with enhanced class balancing!")
    else:
        print("📊 Models trained - test on real car horn audio to verify improvement")

if __name__ == "__main__":
    main()
