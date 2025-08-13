"""
Step 3: Basic Data Cleaning and Validation for Audio based object classification
==================================================================================

This script performs essential data cleaning and validation on the
UrbanSound8K dataset, focusing on creating a clean dataset for
YAMNet training.

Cleaning tasks include:
- Audio file validation and accessibility check
- Basic quality filtering (duration, silence, corruption)
- Creation of cleaned metadata for preprocessing
"""

import os
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

class AudioDataCleaner:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.metadata_path = self.data_dir / "filtered_metadata.csv"
        self.audio_dir = self.data_dir / "UrbanSound8K" / "audio"
        
        # Basic quality thresholds
        self.quality_thresholds = {
            'min_duration': 0.5,      
            'max_duration': 10.0,     
            'min_rms_energy': 0.001, 
            'max_silence_ratio': 0.9  
        }
        
    def load_metadata(self):
        """Load filtered metadata"""
        print("📊 Loading Metadata for Cleaning...")
        
        if not self.metadata_path.exists():
            print("❌ Filtered metadata not found! Run previous steps first.")
            return None
        
        df = pd.read_csv(self.metadata_path)
        print(f"✅ Loaded {len(df)} samples for cleaning")
        
        return df
    
    def validate_audio_files(self, df):
        """Basic audio file validation"""
        print("\n🔍 Validating Audio Files...")
        
        valid_files = []
        invalid_files = []
        
        for idx, row in df.iterrows():
            audio_path = self.audio_dir / f"fold{row['fold']}" / row['slice_file_name']
            
            try:
                # Check if file exists and is readable
                if not audio_path.exists():
                    invalid_files.append(f"Missing: {row['slice_file_name']}")
                    continue
                
                # Try to load audio file
                y, sr = librosa.load(audio_path, sr=None, duration=1.0)  # Just load 1 second for validation
                
                if len(y) == 0 or sr is None:
                    invalid_files.append(f"Corrupted: {row['slice_file_name']}")
                    continue
                
                valid_files.append(idx)
                
            except Exception as e:
                invalid_files.append(f"Error: {row['slice_file_name']} - {str(e)}")
                continue
        
        # Filter to valid files only
        valid_df = df.iloc[valid_files].copy()
        
        print(f"📊 Validation Results:")
        print(f"   Valid files: {len(valid_files)}")
        print(f"   Invalid files: {len(invalid_files)}")
        
        if invalid_files:
            print("⚠️  Invalid files found:")
            for invalid in invalid_files[:5]:  # Show first 5
                print(f"     {invalid}")
            if len(invalid_files) > 5:
                print(f"     ... and {len(invalid_files) - 5} more")
        
        return valid_df
    
    def basic_quality_check(self, df):
        """Basic quality assessment"""
        print("\n🔍 Performing Basic Quality Checks...")
        
        quality_results = []
        
        for idx, row in df.iterrows():
            audio_path = self.audio_dir / f"fold{row['fold']}" / row['slice_file_name']
            
            try:
                # Load audio
                y, sr = librosa.load(audio_path, sr=None)
                
                # Basic quality metrics
                duration = len(y) / sr
                rms_energy = np.sqrt(np.mean(y**2))
                
                # Simple silence detection
                silence_threshold = np.max(np.abs(y)) * 0.01  # 1% of max amplitude
                silent_samples = np.sum(np.abs(y) < silence_threshold)
                silence_ratio = silent_samples / len(y)
                
                # Quality checks
                duration_ok = self.quality_thresholds['min_duration'] <= duration <= self.quality_thresholds['max_duration']
                energy_ok = rms_energy >= self.quality_thresholds['min_rms_energy']
                silence_ok = silence_ratio <= self.quality_thresholds['max_silence_ratio']
                
                quality_results.append({
                    'index': idx,
                    'file': row['slice_file_name'],
                    'class': row['class'],
                    'duration': duration,
                    'rms_energy': rms_energy,
                    'silence_ratio': silence_ratio,
                    'duration_ok': duration_ok,
                    'energy_ok': energy_ok,
                    'silence_ok': silence_ok,
                    'quality_ok': duration_ok and energy_ok and silence_ok
                })
                
            except Exception as e:
                print(f"   ⚠️  Error analyzing {audio_path}: {e}")
                continue
        
        quality_df = pd.DataFrame(quality_results)
        
        # Quality summary
        total_samples = len(quality_df)
        good_quality = len(quality_df[quality_df['quality_ok']])
        
        print(f"📊 Quality Assessment Results:")
        print(f"   Total analyzed: {total_samples}")
        print(f"   Good quality: {good_quality} ({good_quality/total_samples*100:.1f}%)")
        print(f"   Duration issues: {len(quality_df[~quality_df['duration_ok']])}")
        print(f"   Energy issues: {len(quality_df[~quality_df['energy_ok']])}")
        print(f"   Silence issues: {len(quality_df[~quality_df['silence_ok']])}")
        
        return quality_df
    
    def create_cleaned_dataset(self, df, quality_df):
        """Create final cleaned dataset"""
        print("\n🧹 Creating Cleaned Dataset...")
        
        # Get indices of good quality samples
        good_indices = quality_df[quality_df['quality_ok']]['index'].tolist()
        
        # Filter original dataframe
        cleaned_df = df.iloc[good_indices].copy().reset_index(drop=True)
        
        # Summary statistics
        original_count = len(df)
        cleaned_count = len(cleaned_df)
        removal_rate = (original_count - cleaned_count) / original_count * 100
        
        print(f"📊 Cleaning Summary:")
        print(f"   Original samples: {original_count}")
        print(f"   Cleaned samples: {cleaned_count}")
        print(f"   Removed samples: {original_count - cleaned_count} ({removal_rate:.1f}%)")
        
        # Class-wise summary
        print(f"\n📋 Class-wise Results:")
        print(f"{'Class':15} {'Original':>8} {'Cleaned':>8} {'Retention':>10}")
        print("-" * 50)
        
        for class_name in df['class'].unique():
            original_class_count = len(df[df['class'] == class_name])
            cleaned_class_count = len(cleaned_df[cleaned_df['class'] == class_name])
            retention_rate = cleaned_class_count / original_class_count * 100
            
            print(f"{class_name:15} {original_class_count:8d} {cleaned_class_count:8d} {retention_rate:9.1f}%")
        
        # Save cleaned dataset
        cleaned_path = self.data_dir / "cleaned_metadata.csv"
        cleaned_df.to_csv(cleaned_path, index=False)
        
        print(f"\n✅ Cleaned dataset saved to {cleaned_path}")
        
        return cleaned_df

def main():
    """Main cleaning function"""
    print("🧹 Basic Data Cleaning for Audio Classification")
    print("=" * 60)
    
    try:
        # Initialize cleaner
        cleaner = AudioDataCleaner()
        
        # Load metadata
        df = cleaner.load_metadata()
        if df is None:
            return
        
        # Validate audio files
        valid_df = cleaner.validate_audio_files(df)
        
        # Perform quality checks
        quality_df = cleaner.basic_quality_check(valid_df)
        
        # Create cleaned dataset
        cleaned_df = cleaner.create_cleaned_dataset(valid_df, quality_df)
        
        print("\n✅ Data Cleaning Complete!")
        print(f"\n📋 Summary:")
        print(f"   - {len(df)} original samples")
        print(f"   - {len(valid_df)} valid files")
        print(f"   - {len(cleaned_df)} final cleaned samples")
        print(f"   - cleaned_metadata.csv created")
        
        print("\n📋 Next Steps:")
        print("   - Run 04_data_preprocessing.py for audio preprocessing")
        print("   - Proceed with YAMNet feature extraction")
        
    except Exception as e:
        print(f"❌ Error during data cleaning: {str(e)}")
        raise

if __name__ == "__main__":
    main()
