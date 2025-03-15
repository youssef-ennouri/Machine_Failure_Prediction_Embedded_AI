# Machine Failure Prediction Embedded AI

## Project Overview
This project implements a deep learning solution for predictive maintenance in industrial machinery, designed specifically for deployment on STM32 microcontrollers. The system analyzes sensor data to predict machine failures before they occur, allowing for timely maintenance interventions and reduced downtime.

## Dataset
The project utilizes the AI4I 2020 Predictive Maintenance Dataset, which contains 10,000 instances of industrial sensor data. Each data point represents the operating condition of a machine and includes sensor readings such as:
- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]

The dataset classifies failures into five categories:
- TWF: Tool Wear Failure
- HDF: Heat Dissipation Failure
- PWF: Power Failure
- OSF: Overstrain Failure
- RNF: Random Failure

## Technical Implementation

### Data Preprocessing
- Feature standardization using StandardScaler
- Handling class imbalance with SMOTE (Synthetic Minority Over-sampling Technique)
- Strategic train-test split with stratification to maintain class distributions

### Model Architecture
The neural network architecture consists of:
- Input layer matching sensor data dimensionality
- Multiple dense layers with ReLU activation
- Batch normalization layers for training stability
- Dropout layers to prevent overfitting
- Sigmoid activation in the output layer for multi-label classification

### Training Process
- Binary cross-entropy loss function
- Adam optimizer with learning rate scheduling
- Early stopping to prevent overfitting
- Class weights to address imbalanced failure types

### Performance Evaluation
- Binary accuracy metrics
- Precision and recall for each failure type
- Confusion matrices to visualize true vs. predicted failures
- Classification reports for detailed performance analysis

## Repository Structure
- `/data`: Dataset storage and preprocessing scripts
- `/models`: Model architecture definitions and trained weights
- `/deployment`: STM32 deployment code and optimization scripts
- `/evaluation`: Performance metrics and visualization tools
- `/docs`: Additional documentation and implementation guides
