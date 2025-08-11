# Disease Detection from Chest X-Rays

## Project Overview

This project leverages deep learning techniques to detect various lung diseases from chest X-ray images. By employing convolutional neural networks (CNNs) and transfer learning with models like MobileNetV2, the system aims to assist healthcare professionals in diagnosing conditions such as pneumonia, tuberculosis, and COVID-19.

## Dataset

The model is trained on the **ChestX-ray8** dataset, which comprises 108,948 frontal-view X-ray images from 32,717 unique patients. Each image is labeled with up to 14 different thoracic diseases, including conditions like Cardiomegaly, Mass, Pneumothorax, and Edema.
Dataset source: [Chest X-Ray Images (Pneumonia) by Paul Mooney on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## Model Architecture

We utilize a **Sequential CNN model** followed by a **MobileNetV2-based transfer learning model**:

### Sequential CNN Model

- Input: 150x150 RGB images  
- Layers: Conv2D, MaxPooling2D, Flatten, Dense 
- Output: 2 classes (NORMAL vs. PNEUMONIA)

### MobileNetV2 Transfer Learning Model

- Base: MobileNetV2 pre-trained on ImageNet  
- Input: 150x150 RGB images  
- Layers: GlobalAveragePooling2D, Dense  
- Output: 2 classes (NORMAL vs. PNEUMONIA)

## Training Process

- **Epochs:** 10  
- **Optimizer:** Adam  
- **Loss Function:** SparseCategoricalCrossentropy  
- **Metrics:** Accuracy

The training involves fine-tuning the MobileNetV2 model on the ChestX-ray8 dataset, adjusting the final layers to classify between diseased and healthy states.

## Evaluation Metrics

### Accuracy, Precision, Recall, F1-Score

- **Accuracy:** Measures the overall correctness of the model.  
- **Precision:** Indicates the proportion of positive predictions that are actually correct.  
- **Recall:** Represents the proportion of actual positives correctly identified.  
- **F1-Score:** The harmonic mean of precision and recall, providing a balance between the two.

### Confusion Matrix

A confusion matrix is used to visualize the performance of the classification model, showing the true positives, true negatives, false positives, and false negatives.

## Visualizations

### Training vs. Validation Performance

- **Loss Curves:** Plot training and validation loss over epochs to assess overfitting or underfitting.  
- **Accuracy Curves:** Compare training and validation accuracy to evaluate model generalization.

### Grad-CAM Visualizations

Using Grad-CAM to highlight the regions of the X-ray images that the model focuses on during prediction, aiding in interpretability.

## Results Comparison

| Model Type                  | Final Accuracy | Final Loss |
|-----------------------------|----------------|------------|
| Sequential CNN              | 93.95%         | 0.1630     |
| MobileNetV2 Transfer Learning | 91.66%       | 0.2030     |

*Note: The Sequential CNN model achieved higher accuracy and lower loss compared to the MobileNetV2-based model.*


