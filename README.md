# YBI-Foundation-4-Month-Internship
LSTM Based Cyclone Sensors Anomaly Detection System
# Overview
This project implements an LSTM (Long Short-Term Memory) autoencoder for detecting anomalies in cyclone sensor data. The model learns normal patterns in multivariate time series data and identifies unusual observations by analyzing reconstruction errors.

## Project Description
The notebook demonstrates a deep learning approach to anomaly detection using:
- **LSTM Autoencoder Architecture**: Captures temporal dependencies in sensor readings
- **Reconstruction Error Analysis**: Identifies anomalies based on deviation from learned patterns
- **Unsupervised Learning**: No labeled anomaly data required for training
- **Time Series Processing**: Handles sequential sensor data with multiple features


## Requirements
```
pandas
numpy
matplotlib
seaborn
tensorflow/keras
scikit-learn
scipy
```

## Installation
```bash
pip install pandas numpy matplotlib seaborn tensorflow scikit-learn scipy
```

## Notebook Structure

### 1. **Library Import**
- Imports all required libraries and sets random seeds for reproducibility
- Configures plotting styles and suppresses warnings

### 2. **Data Loading**
- Loads cyclone sensor data from Excel file
- Displays initial data shape and time range
- Extracts feature columns (all except 'time')

### 3. **Data Preprocessing**
- Converts columns to numeric type
- Detects and reports missing values
- Removes duplicate timestamps
- Sorts data by time chronologically
- Provides summary statistics

### 4. **Handling Missing Values**
- Uses forward fill followed by backward fill for imputation
- Removes any remaining missing values if necessary
- Validates data quality after preprocessing

### 5. **Input Pipeline Construction**
- Scales features using MinMaxScaler (range: 0-1)
- Creates sequences of length 12 (12 timesteps per sequence)
- Splits data into training (80%) and validation (20%) sets
- Prepares data for LSTM model

### 6. **LSTM Autoencoder Architecture**
```
Input Layer → LSTM(128) → Dropout(0.2)
           → LSTM(64) → Dropout(0.2)
           → Dense(32) [Bottleneck]
           → RepeatVector(12)
           → LSTM(64) → Dropout(0.2)
           → LSTM(128) → Dropout(0.2)
           → TimeDistributed(Dense(n_features))
           → Output Layer
```

### 7. **Training Configuration**
- **Epochs**: 50
- **Batch Size**: 32
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error (MSE)
- **Early Stopping**: Patience of 10 epochs on validation loss

### 8. **Training Process**
- Trains the autoencoder to minimize reconstruction error on normal data
- Monitors both training and validation loss
- Stops early if validation loss doesn't improve

### 9. **Computing Reconstruction Errors**
- Generates predictions for all sequences
- Calculates Mean Squared Error between original and reconstructed sequences
- Provides statistics (min, max, mean, std) of reconstruction errors

### 10. **Anomaly Detection**
- Sets anomaly threshold at 95th percentile of reconstruction errors
- Identifies sequences exceeding the threshold as anomalies
- Reports anomaly count and anomaly rate

### 11. **Results Analysis**
- Lists top 10 sequences with highest reconstruction errors
- Provides detailed error values for investigation

## Usage

### Basic Workflow
1. Prepare your cyclone sensor data in Excel format with a 'time' column and feature columns
2. Update the file path in the "Load Dataset" cell
3. Run all cells sequentially from top to bottom
4. Review anomaly detection results and reconstructed errors

### Customization
You can adjust key parameters in the "Input Pipeline Construction" cell:
```python
sequence_length = 12      # Number of timesteps in each sequence
test_size = 0.2           # Validation set proportion (20%)
percentile = 95           # Percentile for anomaly threshold
epochs = 50               # Training epochs
batch_size = 32           # Batch size for training
```

## Model Architecture Details

### Encoder
- Compresses multivariate time series into a 32-dimensional bottleneck
- Layer 1: LSTM(128) with dropout - learns complex temporal patterns
- Layer 2: LSTM(64) with dropout - abstracts patterns further
- Dense(32) - creates compact representation

### Decoder
- Reconstructs sequences from bottleneck representation
- RepeatVector - repeats encoded representation for each timestep
- Layer 1: LSTM(64) with dropout - begins reconstruction
- Layer 2: LSTM(128) with dropout - recovers temporal structure
- TimeDistributed Dense - outputs all features for each timestep

### Why Dropout?
- Prevents overfitting during training
- Improves generalization to unseen data
- Enhances robust feature learning

## Output Interpretation

### Reconstruction Errors
- **Normal Data**: Low reconstruction errors (model fits well)
- **Anomalies**: High reconstruction errors (model deviates significantly)
- **Threshold**: 95th percentile separates normal from anomalous

### Key Metrics
```
Min error          : Lowest reconstruction error (best fit)
Max error          : Highest reconstruction error (strongest anomaly)
Mean error         : Average model performance
Std error          : Variability in reconstruction errors
Anomaly rate       : Percentage of data flagged as anomalous
```

## Data Requirements

### Format
- **File Format**: Excel (.xlsx)
- **Required Column**: 'time' (timestamp column)
- **Feature Columns**: All numeric columns except 'time'
- **Sample Size**: Minimum ~100+ sequences recommended

### Data Characteristics
- Multivariate time series data
- Regular time intervals preferred
- Numeric features only
- Handle missing values before or during preprocessing

## Advantages of This Approach

1. **Unsupervised Learning**: No labeled anomaly data needed
2. **Captures Temporal Dependencies**: LSTM handles sequential patterns
3. **Multivariate Analysis**: Processes multiple sensor readings simultaneously
4. **Adaptive Threshold**: 95th percentile automatically adjusts to data distribution
5. **Interpretable**: Reconstruction errors provide clear anomaly scores
6. **Scalable**: Efficient for large datasets with multiple features

## Limitations

- Assumes normal behavior is better learned than anomalies
- Performance depends on sequence length selection
- Requires sufficient normal data for training
- Sensitive to input scaling
- May flag rare-but-normal patterns as anomalies
