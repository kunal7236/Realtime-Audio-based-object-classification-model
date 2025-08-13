# Real-Time Audio-Based Object Classification

## Project Overview

This project implements a real-time audio classification pipeline that identifies and classifies environmental sounds using machine learning. The system is designed for robust, real-time detection of critical sound events (e.g., car horn, siren, construction noise) using YAMNet feature extraction and advanced classifiers. The pipeline is suitable for safety, monitoring, and smart environment applications.

## Target Classes for Safety Alert System

Based on road safety and common dangerous scenarios for visually impaired individuals:

### Critical Safety Classes:

1. **car_horn** - Vehicle approaching/warning
2. **engine_idling** - Vehicle nearby (parked or moving)
3. **siren** - Emergency vehicles (ambulance, police, fire truck)
4. **drilling** - Construction/road work
5. **jackhammer** - Heavy construction work
6. **motorcycle** - Fast-moving vehicle
7. **street_music** - Indicates pedestrian areas
8. **children_playing** - School zones/residential areas
9. **dog_bark** - Potential animal hazard
10. **air_conditioner** - Building/indoor proximity indicator

These classes help identify:

- Vehicle proximity and movement
- Construction zones and hazards
- Emergency situations
- Safe vs unsafe areas for navigation
- Environmental context for better decision making

## Project Structure & Script Descriptions

```
├── 01_dataset_acquisition.py         # Download and prepare UrbanSound8K dataset and metadata
├── 02_data_exploration_simplified.py # Explore and visualize dataset class distribution
├── 03_data_cleaning_simplified.py    # Clean and validate metadata, remove unusable samples
├── 04_data_preprocessing_simplified.py # Split data into train/validation/test for YAMNet pipeline
├── 05_yamnet_training_enhanced.py    # Extract YAMNet features, train Random Forest & Neural Network with class balancing
├── 06_yamnet_validation.py           # Validate models on unseen validation set
├── 07_yamnet_testing.py              # Final unbiased testing and model selection
├── 08_simple_yamnet_realtime.py      # Real-time audio classification using trained models
├── simple_yamnet_realtime.py         # Alternate real-time classification script
├── test_audio_file.py                # Test classification on a single audio file
├── evaluate_yamnet_models.py         # Compare and evaluate trained models
├── requirements.txt                  # Python dependencies
└── models/                           # Saved models (Random Forest, Neural Network, Label Encoder)
```

### Pipeline Overview

- **01_dataset_acquisition.py**: Downloads UrbanSound8K, extracts and formats metadata for downstream processing.
- **02_data_exploration_simplified.py**: Analyzes class distribution and dataset characteristics.
- **03_data_cleaning_simplified.py**: Cleans metadata, removes missing/corrupt files, ensures data quality.
- **04_data_preprocessing_simplified.py**: Splits data into train/validation/test sets for fair evaluation.
- **05_yamnet_training_enhanced.py**: Extracts YAMNet embeddings from audio, trains Random Forest and Neural Network models with aggressive class balancing for rare events.
- **06_yamnet_validation.py**: Validates models on the validation set, tunes hyperparameters, and prevents overfitting.
- **07_yamnet_testing.py**: Performs final evaluation on the test set for unbiased performance reporting.
- **08_simple_yamnet_realtime.py** & **simple_yamnet_realtime.py**: Real-time audio classification using microphone input and trained models.
- **test_audio_file.py**: Classifies a single audio file using trained models.
- **evaluate_yamnet_models.py**: Compares model performance and generates evaluation reports.

## Dataset Split

- Training: 70%
- Validation: 20%
- Testing: 10%

## Key Features

- Real-time audio classification using YAMNet embeddings
- Enhanced class balancing for rare and safety-critical events
- Dual model approach: Random Forest and Deep Neural Network classifiers
- Modular scripts for each pipeline step (acquisition, cleaning, training, testing, real-time)
- Performance monitoring and evaluation
- Easily extensible for new sound classes or environments

## Technical Implementation

### Feature Extraction

The system uses Google's YAMNet model from TensorFlow Hub to extract rich audio embeddings. These embeddings capture complex audio patterns that are then used for classification.

### Model Training

- **Random Forest Classifier**: Implemented with enhanced parameters (300 estimators, depth 25) and class weighting to boost detection of rare but critical classes.
- **Neural Network**: Multi-layer network with batch normalization, dropout for regularization, and early stopping to prevent overfitting.
- **Class Weighting**: Safety-critical classes like "car_horn" and "siren" receive 2.0x weight boost, moderately rare classes receive 1.5x boost.

### Real-time Processing

- Threaded audio capture for continuous monitoring
- Rolling buffer for smooth classification
- Confidence thresholds to reduce false alarms

## Dataset Setup

This project uses the UrbanSound8K dataset, which is not included in the repository due to its size.

### Downloading the Dataset

1. Download the UrbanSound8K dataset from the official source:

   - https://urbansounddataset.weebly.com/urbansound8k.html
   - You will need to complete a form to access the download link

2. Place the dataset in the project structure:
   - Extract the downloaded ZIP file
   - Place the entire `UrbanSound8K` folder in the `data/` directory
   - Final path should be: `data/UrbanSound8K/`

### Expected Structure

```
data/
  UrbanSound8K/
    audio/
      fold1/
      fold2/
      ...
      fold10/
    metadata/
      UrbanSound8K.csv
```

Alternatively, you can run `01_dataset_acquisition.py` which will attempt to download and set up the dataset automatically (if available).

## Requirements

See `requirements.txt` for all dependencies. Install with:

```bash
pip install -r requirements.txt
```
