"""
Step 1: Dataset Acquisition for Realtime audio based object detection
==================================================================

This script downloads and sets up the UrbanSound8K dataset

"""

import os
import sys
import requests
import tarfile
import pandas as pd
import numpy as np
from pathlib import Path
import time
import psutil
from memory_profiler import profile

# System monitoring
def get_system_stats():
    """Get current system resource usage"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    return {
        'cpu_percent': cpu_percent,
        'memory_used_gb': memory.used / (1024**3),
        'memory_percent': memory.percent
    }

class UrbanSoundDatasetManager:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.target_classes = {
            'car_horn': 1,          
            'engine_idling': 2,    
            'siren': 3,            
            'drilling': 4,          
            'jackhammer': 5,       
            'street_music': 6,     
            'children_playing': 7,  
            'dog_bark': 8,         
            'air_conditioner': 9    
        }
        
        self.metadata_path = self.data_dir / "UrbanSound8K" / "metadata" / "UrbanSound8K.csv"
        
    @profile
    def download_dataset(self):
        """Download UrbanSound8K dataset"""
        print("📥 Starting UrbanSound8K Dataset Download...")
        print("⚠️  Note: This dataset is large (~6GB) and requires registration")
        print("📋 Please download manually from: https://urbansounddataset.weebly.com/urbansound8k.html")
        print("\n🎯 Target Classes for Visually Impaired Safety:")
        
        for class_name, class_id in self.target_classes.items():
            print(f"   {class_id:2d}. {class_name.replace('_', ' ').title()}")
        
        # Check if dataset already exists
        if self.metadata_path.exists():
            print("✅ Dataset already exists!")
            return True
        
        print("\n📁 Expected dataset structure after download:")
        print("data/")
        print("└── UrbanSound8K/")
        print("    ├── audio/")
        print("    │   ├── fold1/ ... fold10/")
        print("    └── metadata/")
        print("        └── UrbanSound8K.csv")
        
        return False
    
    def verify_dataset(self):
        """Verify dataset integrity and structure"""
        print("\n🔍 Verifying Dataset Structure...")
        
        if not self.metadata_path.exists():
            print("❌ Metadata file not found!")
            return False
        
        # Load metadata
        df = pd.read_csv(self.metadata_path)
        print(f"✅ Metadata loaded: {len(df)} samples")
        
        # Check audio folders
        audio_dir = self.data_dir / "UrbanSound8K" / "audio"
        fold_count = 0
        total_files = 0
        
        for fold in range(1, 11):
            fold_dir = audio_dir / f"fold{fold}"
            if fold_dir.exists():
                file_count = len(list(fold_dir.glob("*.wav")))
                total_files += file_count
                fold_count += 1
                print(f"   Fold {fold}: {file_count} audio files")
        
        print(f"✅ Found {fold_count}/10 folds with {total_files} total audio files")
        
        # Verify target classes exist
        available_classes = df['class'].unique()
        missing_classes = []
        available_target_classes = []
        
        for class_name in self.target_classes.keys():
            if class_name in available_classes:
                available_target_classes.append(class_name)
                count = len(df[df['class'] == class_name])
                print(f"   ✅ {class_name}: {count} samples")
            else:
                missing_classes.append(class_name)
        
        if missing_classes:
            print(f"⚠️  Missing target classes: {missing_classes}")
        
        print(f"📊 Available target classes: {len(available_target_classes)}/{len(self.target_classes)}")
        
        return len(available_target_classes) > 0
    
    def analyze_class_distribution(self):
        """Analyze distribution of target classes"""
        print("\n📊 Analyzing Class Distribution for Safety Classes...")
        
        if not self.metadata_path.exists():
            print("❌ Metadata file not found!")
            return None
        
        df = pd.read_csv(self.metadata_path)
        
        # Filter for target classes only
        target_df = df[df['class'].isin(self.target_classes.keys())].copy()
        
        if len(target_df) == 0:
            print("❌ No target classes found in dataset!")
            return None
        
        print(f"📈 Total samples in target classes: {len(target_df)}")
        
        # Class distribution
        class_counts = target_df['class'].value_counts()
        print("\n📋 Class Distribution:")
        
        for class_name, count in class_counts.items():
            percentage = (count / len(target_df)) * 100
            print(f"   {class_name:15} | {count:4d} samples ({percentage:5.1f}%)")
        
        # Fold distribution
        print("\n📁 Fold Distribution:")
        fold_dist = target_df['fold'].value_counts().sort_index()
        for fold, count in fold_dist.items():
            print(f"   Fold {fold:2d}: {count:4d} samples")
        
        return target_df
    
    def create_filtered_metadata(self):
        """Create filtered metadata with only target classes"""
        print("\n🎯 Creating Filtered Metadata for Target Classes...")
        
        if not self.metadata_path.exists():
            print("❌ Metadata file not found!")
            return None
        
        # Load original metadata
        df = pd.read_csv(self.metadata_path)
        
        # Filter for target classes
        filtered_df = df[df['class'].isin(self.target_classes.keys())].copy()
        
        if len(filtered_df) == 0:
            print("❌ No target classes found!")
            return None
        
        # Add numeric class IDs for our target classes
        filtered_df['target_class_id'] = filtered_df['class'].map(self.target_classes)
        
        # Save filtered metadata
        filtered_path = self.data_dir / "filtered_metadata.csv"
        filtered_df.to_csv(filtered_path, index=False)
        
        print(f"✅ Filtered metadata saved: {filtered_path}")
        print(f"📊 Filtered dataset: {len(filtered_df)} samples across {len(self.target_classes)} safety classes")
        
        return filtered_df

def main():
    """Main execution function"""
    print("🎵 Audio-Based Object Identification for Visually Impaired Assistance")
    print("=" * 70)
    print("Step 1: Dataset Acquisition and Verification")
    print("=" * 70)
    
    # Monitor system resources
    start_time = time.time()
    start_stats = get_system_stats()
    
    # Initialize dataset manager
    dataset_manager = UrbanSoundDatasetManager()
    
    try:
        # Step 1: Download/verify dataset
        download_success = dataset_manager.download_dataset()
        
        if not download_success:
            print("\n⏳ Waiting for manual dataset download...")
            print("Please download UrbanSound8K and extract to 'data/' directory")
            return
        
        # Step 2: Verify dataset structure
        if not dataset_manager.verify_dataset():
            print("❌ Dataset verification failed!")
            return
        
        # Step 3: Analyze class distribution
        target_df = dataset_manager.analyze_class_distribution()
        
        if target_df is None:
            print("❌ Class analysis failed!")
            return
        
        # Step 4: Create filtered metadata
        filtered_df = dataset_manager.create_filtered_metadata()
        
        if filtered_df is None:
            print("❌ Metadata filtering failed!")
            return
        
        # System resource summary
        end_time = time.time()
        end_stats = get_system_stats()
        
        print("\n📊 System Performance Summary:")
        print("=" * 40)
        print(f"⏱️  Execution Time: {end_time - start_time:.2f} seconds")
        print(f"💾 Memory Usage: {end_stats['memory_used_gb']:.2f} GB ({end_stats['memory_percent']:.1f}%)")
        print(f"⚡ CPU Usage: {end_stats['cpu_percent']:.1f}%")
        
        print("\n✅ Dataset Acquisition Complete!")
        print("📋 Next Steps:")
        print("   - Run 02_data_exploration.py for detailed analysis")
        print("   - Proceed with data cleaning and preprocessing")
        
    except Exception as e:
        print(f"❌ Error during dataset acquisition: {str(e)}")
        raise

if __name__ == "__main__":
    main()
