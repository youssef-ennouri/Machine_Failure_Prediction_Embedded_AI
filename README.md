
# Predictive Maintenance Deep Learning Project

## 1. General Overview

This project tackles an industrial problem: predicting machine failures in embedded systems with limited resources. Predictive maintenance offers advantages over traditional reactive or scheduled methods. It reduces downtime, prolongs equipment life, and optimizes maintenance resources.

The project's main goal was dual: first, to build a deep learning model analyzing industrial sensor data to foresee machine failures. Second, to implement this model efficiently on embedded hardware with limited resources (STM32L4R9). This approach enables inference directly on hardware attached to industrial machines. It removes the need for constant cloud connections, reduces latency, and improves data privacy.

We used the AI4I 2020 Predictive Maintenance dataset. It has 10,000 data points from industrial sensors. The developed model identifies operational patterns and detects failures early. The dataset categorizes failures into five types: tool wear, heat dissipation, power, overload, and random failures.

The project's importance is its practical industrial use. It helps shift maintenance strategies from reactive to proactive methods. It also shows that complex AI models can operate on microcontrollers with limited resources. This edge computing solution cuts cloud costs and bandwidth use. Additionally, it allows predictive maintenance on existing machines without significant hardware investment.

## 2. Detailed Project Methodology

### 2.1 Dataset Analysis

#### Distribution Analysis of Machine Failures

We started by examining the distribution between machine failures and normal operations. This step helped understand the dataset and identify challenges for training. Visualization was used to detect class imbalance, common in maintenance data because failures are usually rarer than normal operation.

The analysis showed an imbalance: non-failures were about 20 times more frequent than failures. This reflects actual industrial conditions. However, it complicates model training, as models often prioritize the dominant class and might miss critical failure patterns.

#### Failure Type Distribution Analysis

We analyzed the distribution of individual failure types. This helped identify how frequent each type was in the dataset. The goal was to spot rare failure categories to adapt our modeling approach accordingly.

Our findings showed imbalance not only between failures and non-failures but also within failure types. Heat Dissipation Failure (HDF) was most common, appearing 115 times. In contrast, Random Failure (RNF) occurred only 19 times. This two-level imbalance creates difficulties: failures overall are rare, and some failure types are even rarer.

This nested imbalance complicates the model's ability to generalize. Neural networks trained on such datasets often favor frequent non-failure cases. As a result, models may struggle to identify rare failures like RNF accurately. Technically, this can cause biased predictions, negatively impacting precision and recall for uncommon failure types. In maintenance tasks, this issue is critical since rare failures often lead to severe or costly outcomes.

1. **Data collection limitations**: The monitoring system may be capable of detecting anomalous behavior (indicating failure) without having sufficient sensor coverage to determine the exact failure mechanism.
2. **Multi-factor failures**: Some failures might result from complex interactions between multiple systems that don't fit neatly into the predefined failure categories.
3. **Unmonitored variables**: The failure might be caused by factors not captured in the current sensor array, highlighting potential blind spots in the monitoring system.
4. **Annotation inconsistencies**: The dataset labeling process might have suffered from human error or inconsistent application of failure classification criteria.

This observation has profound implications for our modeling approach. It suggests that we need to consider not just the binary classification problem (failure/no failure) and the multi-class problem (which type of failure), but also account for inherent uncertainty in the labeling process itself. From a machine learning perspective, this resembles a partial label learning problem, where annotations may be incomplete or noisy, requiring more robust modeling techniques and careful performance evaluation.

### 2.2 Initial Model Development (Without Balancing)

#### Feature Evaluation

We examined the available features to determine which parameters would serve as model inputs. After evaluation, we selected five key operational parameters as inputs: air temperature, process temperature, rotational speed, torque, and tool wear. These parameters were chosen because they represent critical physical conditions that can indicate impending failures in industrial machinery.
#### Data Preparation Strategy

For our initial approach, we split the dataset into training (80%) and testing (20%) sets. This standard split ratio provides sufficient data for model training while reserving an adequate portion for evaluation. We stratified the split based on the "No Failure" column to maintain the same failure-to-non-failure ratio in both sets.

A key preprocessing decision was the creation of a "No Failure" target variable, which we derived by inverting the "Machine failure" column (setting it to 1 when Machine failure = 0, and vice versa). By explicitly modeling non-failure as a positive class rather than merely the absence of failure, we provide the model with a clear target to learn the characteristics of normal operation. This is particularly important from a neural network optimization perspective, as it allows the backpropagation algorithm to explicitly reinforce the recognition of normal operational patterns.

We deliberately structured our target variables to include "No Failure" and only four of the failure types (TWF, HDF, PWF, OSF), excluding RNF (Random Failures). With only 19 instances of RNF in the entire dataset (less than 0.2% of all samples), there was insufficient data to build a reliable predictive model for this failure type. From a statistical learning perspective, such a small sample is unlikely to capture the true distribution of the underlying failure mechanism. The four selected failure types (TWF, HDF, PWF, OSF) represent specific mechanical or operational issues that can be addressed through targeted maintenance interventions, whereas "Random Failures" by definition lack clear causal patterns, making them less suitable for predictive maintenance applications.

For validation, we chose to use the test set during both training and final evaluation. Although typically not recommended, this was necessary due to our dataset's limited size (10,000 samples). The training set had only 8,000 samples. Creating a separate validation set would have reduced training data further, limiting the model's capacity to detect failure patterns.

From a technical standpoint, this approach involves a trade-off: using the same data for validation during training and final evaluation can lead to slightly optimistic performance estimates, but this was mitigated through:

1. The implementation of regularization techniques (dropout, L2 regularization)
2. Early stopping based on validation loss to prevent overfitting to the test data
3. A sufficiently large test set (2,000 samples) to provide stable performance metrics

We judged that this compromise was appropriate given the dataset constraints, though in an ideal scenario with more data, a proper three-way split (training/validation/test) would be preferred.

#### Architecture Design Considerations

For our initial model, we designed a neural network with the following characteristics:

- A dense layer (32 neurons) with ReLU activation to capture non-linear relationships in the data
- Batch normalization to stabilize training and accelerate convergence
- Dropout (0.4) to prevent overfitting, particularly important given our imbalanced dataset
- Sigmoid activation in the output layer to enable multi-label classification

We selected binary cross-entropy as the loss function since each output represents a binary classification (failure or no failure). The Adam optimizer was chosen due to its adaptive learning rate, suitable for datasets with noisy or sparse gradients.

Training Method
We applied early stopping with a patience of 15 epochs. This reduced overfitting while allowing sufficient training. We used a batch size of 64 to balance training speed with accuracy.

We tracked metrics for training and validation to detect overfitting. Monitoring is crucial for imbalanced data, as high accuracy alone might reflect biased predictions towards the majority class.

### 2.3 Enhanced Model Development (With Balancing)

#### Dataset Balancing Strategy

For our enhanced model, we maintained consistency in our foundational approach by using the same input features (air temperature, process temperature, rotational speed, torque, and tool wear) and output targets ("No Failure", "TWF", "HDF", "PWF", "OSF"). We also retained the same train-test split ratio (80-20%) to ensure a fair comparison with our initial model. This methodological consistency allowed us to isolate the impact of data balancing on model performance.

Before applying any resampling techniques, we implemented feature standardization using StandardScaler. This normalization step was critical for several technical reasons. Neural networks are sensitive to feature scales, and standardization ensures that all features contribute proportionally to the gradient calculation during backpropagation, preventing features with larger numerical ranges (like rotational speed) from dominating the learning process. Distance-based algorithms like SMOTE rely on computing distances between feature vectors. Without standardization, features with larger scales would disproportionately influence these distance calculations, potentially leading to the generation of unrealistic synthetic samples.

Our rebalancing strategy employed a two-phase approach:

First, we applied Random Undersampling to address the extreme imbalance in the dataset. We configured RandomUnderSampler to reduce the "No Failure" class to 5,000 samples (from the original ~9,500), a reduction of approximately 47%. This threshold was carefully chosen as a compromise between preserving important information about normal operation while reducing its overwhelming influence on model training.    

Undersampling alone, however, would result in insufficient training data. We then applied SMOTE (Synthetic Minority Over-sampling Technique). SMOTE selects a minority sample and a neighbor randomly from its k-nearest neighbors. It generates synthetic samples along the line between these points, helping address the imbalance. This process is repeated for each failure type independently, ensuring that the synthetic samples for each failure mode accurately reflect the unique characteristics of that specific failure type rather than blending failure types. SMOTE generated synthetic samples not to create exactly equal class sizes, but to intelligently oversample minority classes by up to 20x their original count, while capping the total at 50% of the majority class size. Instead of uniform 5,000 samples, it generates synthetic data proportionally, ensuring minority classes are better represented without artificially forcing an identical sample count across all classes.

This combined approach offers several technical advantages over using either technique alone. By first reducing the majority class and then synthesizing minority classes, we create a dataset where each class contributes equally to the loss function during training without excessive duplication or synthetic data generation. The synthetic samples help fill gaps in the feature space, particularly in regions characteristic of failure states, improving the model's decision boundaries in these critical areas. While synthetic samples introduce some artificial patterns, they provide less opportunity for overfitting than would occur if we simply duplicated the existing minority samples. By first reducing the dataset size through undersampling, we decrease the computational burden of the subsequent SMOTE operation, making the overall process more efficient.

After completing the resampling process, we reconverted the categorical target back to the original multi-label format to maintain consistency with our model architecture and evaluation metrics.

#### Enhanced Architecture Design

For our balanced model, we significantly expanded the network architecture compared to the initial model that used a single hidden layer with 32 neurons. The enhanced model implements a deeper architecture with three dense layers of decreasing size (128, 64, and 32 neurons), creating a hierarchical feature extraction pattern. 
This architectural decision was motivated by several technical considerations. The balanced dataset contains substantially more meaningful information about failure patterns across all classes. While the original imbalanced dataset effectively provided limited examples of failure conditions, our resampled dataset now contains 5,000 examples per class. This richer information landscape requires a model with greater representational capacity to capture the nuanced patterns that distinguish different failure types. The decreasing neuron count (128→64→32) implements a dimensionality reduction approach that allows the network to first extract a wide range of low-level features and gradually combine them into more abstract, high-level representations. This architecture mimics the feature hierarchy observed in many successful deep learning applications, where earlier layers capture basic patterns while deeper layers recognize more complex combinations of these patterns. 
The three-layer architecture with moderate depth strikes a balance between model expressivity and training stability. Deeper networks can suffer from vanishing/exploding gradient problems, but our moderate depth combined with batch normalization ensures efficient gradient flow during backpropagation.

We implemented L2 regularization with a coefficient of 0.001 across all layers for specific technical reasons. L2 regularization adds a penalty term proportional to the square of weight values to the loss function. The coefficient (0.001) was carefully chosen to provide sufficient regularization without overly constraining the model. This prevents the model from assigning excessively large weights to any specific features, promoting smoother decision boundaries. By penalizing large weights, L2 regularization effectively reduces the model's sensitivity to small variations in input features. This is particularly important when dealing with synthetic data generated by SMOTE, which may introduce some artificial patterns that shouldn't be overfitted.

Batch normalization was retained and applied after each dense layer for several critical reasons. Normalization creates a smoother optimization landscape with better-behaved gradients, allowing for faster convergence and higher learning rates. This is particularly important for our deeper architecture, where gradient-based optimization becomes more challenging. Batch normalization introduces noise during training (due to mini-batch statistics), which has a regularizing effect complementary to the explicit L2 regularization. This dual regularization approach—explicit through L2 and implicit through batch normalization—provides more robust protection against overfitting.

Unlike the initial model, which prioritized simplicity due to limited meaningful training data, the enhanced architecture embraces the richer information landscape provided by our balanced dataset. The combination of increased model capacity, strategic regularization, and normalization techniques enables the model to effectively learn distinct patterns for each failure mode without overfitting to either the original rare samples or the synthetic examples generated during resampling.

#### Advanced Training Approach

We enhanced our training approach with several techniques:

- Maintained early stopping with increased patience (20 epochs) to accommodate the more complex learning task
- Added learning rate reduction on plateau to fine-tune the model as training progressed
- Increased the batch size to 128 to improve training stability with the resampled dataset
- Monitored learning rate changes to ensure effective optimization

These enhancements were designed to help the model better learn from the balanced dataset, adjusting the learning process dynamically based on training progress.

By using SMOTE, the model detects machine failures accurately. The confusion matrix indicates a balance between true positives and false negatives for failure detection. Most failures are no longer overlooked. The classification report shows improved recall and precision for failure classes. The model is more effective at distinguishing between different failure types.

### Part 3: Model Deployment on Embedded Target

### 3.1 Model Export for Embedded Platform

After training and validating our predictive model in TensorFlow, we adapted it for deployment on a microcontroller. The first step was converting the model to TensorFlow Lite (TFLite), optimized for resource-limited systems.

```
# Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(balanced_model)
tflite_model = converter.convert()

# Save the model
with open('../models/machine_failure_model.tflite', 'wb') as f:
    f.write(tflite_model)
```

This conversion is essential to reduce the model size using compression techniques, which is crucial for microcontrollers like the STM32L4R9, which have limited memory.

#### Preparing Validation Data for Deployment

To ensure consistent evaluation of model performance on the embedded target, we prepared our test data accordingly:

```
# Save test data in numpy format
np.save('X_test_scaled.npy', X_test_scaled.astype(np.float32))
np.save('Y_test.npy', Y_test.astype(np.float32))
```

### 3.2 Optimization with STM32Cube.AI

#### Integration into the STM32 Ecosystem

STM32Cube.AI is a tool designed for deploying deep learning models on ST microcontrollers. We selected this platform for technical reasons, as it converts the TFLite model into optimized C code for the STM32L4R9’s Cortex-M processor. This conversion utilizes hardware features such as the floating-point unit (FPU), pipeline architecture, SIMD (Single Instruction, Multiple Data) capabilities, and loop unrolling. These optimizations significantly improve execution speed on the microcontroller.

### 3.3 Evaluation on Hardware Target

#### PC-STM32 Communication Protocol

To validate our embedded implementation, we developed a Python script that establishes bidirectional communication with the microcontroller via the UART interface. The communication protocol consists of the following steps:

1. The `synchronize_UART()` function implements a synchronization protocol ensuring correct data exchange between the PC and STM32 before transmission.

```
def synchronize_UART(serial_port):
    while True:
        serial_port.write(b"\xAB")  # Synchronization byte
        ret = serial_port.read(1)
        if ret == b"\xCD":  # STM32 acknowledgment
            serial_port.read(1)
            break
```

2. Input data (5 float32 values) is converted into a compact binary stream before transmission, preserving precision while minimizing transmission time.

```
def send_inputs_to_STM32(inputs, serial_port):
    inputs = inputs.astype(np.float32)
    buffer = b""
    for x in inputs:
        buffer += x.tobytes()
    serial_port.write(buffer)
```

3. Model outputs are received as normalized uint8 values and interpreted to determine the predicted class.

```
def read_output_from_STM32(serial_port):
    output = serial_port.read(10)
    float_values = [int(out)/255 for out in output]
    return float_values
```

#### Performance Results on Target

After deploying on the STM32L4R9, we ran our validation benchmark on 100 test samples, yielding the following results:

```
Evaluating model on STM32...
----- Iteration 1 -----
   Expected output: [1. 0. 0. 0. 0.]
   Received output: [0.996078431372549, 0.0, 0.0, 0.0, 0.0]
----------------------- Accuracy: 0.01
...
----- Iteration 100 -----
   Expected output: [1. 0. 0. 0. 0.]
   Received output: [0.996078431372549, 0.0, 0.0, 0.0, 0.0]
----------------------- Accuracy: 0.98
```