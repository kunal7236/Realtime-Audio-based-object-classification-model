"""
Step 4: Basic Data Preprocessing for Realtime audio based object detection
========================================================================

This script performs essential data preprocessing for the YAMNet training pipeline:
- Train/Validation/Test split (70/20/10)
- Creation of data splits CSV for YAMNet training

The YAMNet training will handle its own feature extraction directly from audio files.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

class AudioPreprocessor:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.metadata_path = self.data_dir / "cleaned_metadata.csv"
    
    def load_cleaned_metadata(self):
        """Load cleaned metadata"""
        print("📊 Loading Cleaned Metadata...")
        
        if not self.metadata_path.exists():
            print("❌ Cleaned metadata not found! Run 03_data_cleaning.py first.")
            return None
        
        df = pd.read_csv(self.metadata_path)
        print(f"✅ Loaded {len(df)} cleaned samples")
        
        return df
    
    def create_stratified_split(self, df):
        """Create stratified train/validation/test split (70/20/10)"""
        print("\n📊 Creating Stratified Data Split (70/20/10)...")
        
        # First split: 70% train, 30% temp
        train_df, temp_df = train_test_split(
            df, 
            test_size=0.3, 
            stratify=df[['class', 'fold']], 
            random_state=42
        )
        
        # Second split: 20% validation, 10% test from temp (30%)
        val_df, test_df = train_test_split(
            temp_df, 
            test_size=1/3,  # 1/3 of 30% = 10%
            stratify=temp_df[['class', 'fold']], 
            random_state=42
        )
        
        # Add split labels
        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()
        
        train_df['split'] = 'train'
        val_df['split'] = 'validation'
        test_df['split'] = 'test'
        
        # Print split statistics
        print(f"📊 Split Statistics:")
        print(f"   Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
        print(f"   Validation: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
        print(f"   Test: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
        
        # Check class distribution in each split
        print(f"\n📊 Class Distribution by Split:")
        print(f"{'Class':15} {'Train':>6} {'Val':>6} {'Test':>6}")
        print("-" * 42)
        
        for class_name in sorted(df['class'].unique()):
            train_count = len(train_df[train_df['class'] == class_name])
            val_count = len(val_df[val_df['class'] == class_name])
            test_count = len(test_df[test_df['class'] == class_name])
            
            print(f"{class_name:15} {train_count:6d} {val_count:6d} {test_count:6d}")
        
        # Combine all splits with split labels
        splits_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        
        # Keep only essential columns for YAMNet training
        essential_columns = ['slice_file_name', 'fsID', 'start', 'end', 'salience', 'fold', 'classID', 'class', 'split']
        splits_df = splits_df[essential_columns]
        
        # Save data splits file (the ONLY output needed for YAMNet training)
        splits_path = self.data_dir / "data_splits.csv"
        splits_df.to_csv(splits_path, index=False)
        
        print(f"\n✅ Data splits saved to {splits_path}")
        
        return train_df, val_df, test_df

def main():
    """Main preprocessing function"""
    print("🎵 Basic Data Preprocessing for YAMNet Training")
    print("=" * 60)
    
    try:
        # Initialize preprocessor
        preprocessor = AudioPreprocessor()
        
        # Load cleaned metadata
        df = preprocessor.load_cleaned_metadata()
        if df is None:
            return
        
        # Create data splits (the only essential step for YAMNet)
        train_df, val_df, test_df = preprocessor.create_stratified_split(df)
        
        print("\n✅ Data Preprocessing Complete!")
        print(f"\n📋 Summary:")
        print(f"   - {len(train_df)} training samples")
        print(f"   - {len(val_df)} validation samples") 
        print(f"   - {len(test_df)} test samples")
        print(f"   - data_splits.csv created for YAMNet training")
        
        print("\n📋 Next Steps:")
        print("   - Run 05_yamnet_training_enhanced.py for YAMNet-based training")
        print("   - YAMNet will handle its own feature extraction from raw audio files")
        
    except Exception as e:
        print(f"❌ Error during preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    main()
