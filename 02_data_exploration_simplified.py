"""
Step 2: Basic Data Exploration for Audio based object classification
======================================================================

This script performs basic exploratory data analysis (EDA) on the
UrbanSound8K dataset, focusing on essential information for the
YAMNet training pipeline.

Analysis includes:
- Class distribution and balance
- Basic dataset statistics
- File validation
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class AudioDataExplorer:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.metadata_path = self.data_dir / "filtered_metadata.csv"
        self.audio_dir = self.data_dir / "UrbanSound8K" / "audio"
    
    def load_metadata(self):
        """Load and validate filtered metadata"""
        print("📊 Loading Filtered Metadata...")
        
        if not self.metadata_path.exists():
            print("❌ Filtered metadata not found! Run 01_dataset_acquisition.py first.")
            return None
        
        df = pd.read_csv(self.metadata_path)
        print(f"✅ Loaded {len(df)} samples across {df['class'].nunique()} classes")
        
        return df
    
    def analyze_class_distribution(self, df):
        """Analyze class distribution"""
        print("\n📊 Class Distribution Analysis...")
        print("=" * 60)
        
        # Class counts and percentages
        class_counts = df['class'].value_counts().sort_index()
        total_samples = len(df)
        
        print(f"📋 Dataset Overview:")
        print(f"   Total samples: {total_samples}")
        print(f"   Number of classes: {df['class'].nunique()}")
        print(f"   Folds: {sorted(df['fold'].unique())}")
        print()
        
        print(f"📊 Class Distribution:")
        print(f"{'Class':15} {'Count':>6} {'Percentage':>10} {'Adequacy':>12}")
        print("-" * 50)
        
        for class_name in sorted(class_counts.index):
            count = class_counts[class_name]
            percentage = (count / total_samples) * 100
            adequacy = "Good" if count >= 100 else "Limited" if count >= 50 else "Poor"
            
            print(f"{class_name:15} {count:6d} {percentage:9.1f}% {adequacy:>12}")
        
        # Fold distribution
        print(f"\n📁 Fold Distribution:")
        fold_counts = df['fold'].value_counts().sort_index()
        for fold, count in fold_counts.items():
            print(f"   Fold {fold}: {count:3d} samples")
        
        return class_counts
    
    def validate_audio_files(self, df, sample_size=10):
        """Basic validation of audio file accessibility"""
        print(f"\n🔍 Validating Audio File Access (sample: {sample_size})...")
        
        # Sample files from each class
        validation_results = {}
        total_checked = 0
        total_accessible = 0
        
        for class_name in df['class'].unique():
            class_files = df[df['class'] == class_name].sample(min(sample_size, len(df[df['class'] == class_name])))
            
            accessible = 0
            checked = 0
            
            for _, row in class_files.iterrows():
                audio_path = self.audio_dir / f"fold{row['fold']}" / row['slice_file_name']
                checked += 1
                total_checked += 1
                
                if audio_path.exists():
                    accessible += 1
                    total_accessible += 1
            
            validation_results[class_name] = {
                'checked': checked,
                'accessible': accessible,
                'accessibility_rate': (accessible / checked) * 100 if checked > 0 else 0
            }
        
        # Report results
        print(f"📋 File Accessibility Results:")
        print(f"{'Class':15} {'Checked':>8} {'Accessible':>10} {'Rate':>8}")
        print("-" * 50)
        
        for class_name, results in validation_results.items():
            rate = results['accessibility_rate']
            print(f"{class_name:15} {results['checked']:8d} {results['accessible']:10d} {rate:7.1f}%")
        
        overall_rate = (total_accessible / total_checked) * 100 if total_checked > 0 else 0
        print(f"\n📊 Overall Accessibility: {total_accessible}/{total_checked} ({overall_rate:.1f}%)")
        
        if overall_rate < 95:
            print("⚠️  Warning: Some audio files are not accessible!")
        else:
            print("✅ All sampled audio files are accessible")
        
        return validation_results

def main():
    """Main exploration function"""
    print("🎵 Basic Data Exploration for Audio Classification")
    print("=" * 60)
    
    try:
        # Initialize explorer
        explorer = AudioDataExplorer()
        
        # Load metadata
        df = explorer.load_metadata()
        if df is None:
            return
        
        # Analyze class distribution
        class_counts = explorer.analyze_class_distribution(df)
        
        # Validate audio file access
        validation_results = explorer.validate_audio_files(df, sample_size=5)
        
        print("\n✅ Basic Data Exploration Complete!")
        print("\n📋 Summary:")
        print(f"   - {len(df)} total samples")
        print(f"   - {df['class'].nunique()} audio classes")
        print(f"   - {df['fold'].nunique()} data folds")
        print(f"   - Audio files validated")
        
        print("\n📋 Next Steps:")
        print("   - Run 03_data_cleaning.py for data validation and cleaning")
        print("   - Proceed with 04_data_preprocessing.py for feature preparation")
        
    except Exception as e:
        print(f"❌ Error during data exploration: {str(e)}")
        raise

if __name__ == "__main__":
    main()
