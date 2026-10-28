# 🌿 Plant Disease Detection Using CNN

## Project Overview
This project uses a Convolutional Neural Network (CNN) to detect diseases in tomato leaves from images.

The model classifies tomato leaf images into 10 different healthy and disease classes. Image augmentation techniques are also used and compared with a baseline CNN model.

## Features
- Tomato leaf image classification
- 10 disease/healthy classes
- CNN-based deep learning model
- Image augmentation
- Disease prediction with confidence score
- Top-3 predictions
- Disease symptoms and general management advice
- Streamlit web application

## Technologies Used
- Python
- TensorFlow
- Keras
- NumPy
- Streamlit
- Pillow
- Matplotlib
- Scikit-learn

## Dataset
PlantVillage Dataset

Tomato leaf images were selected from the dataset and divided into:
- Training data
- Validation data
- Testing data

## Image Preprocessing
Images are resized to 128 × 128 pixels and pixel values are normalized before being processed by the CNN.

## Image Augmentation
The following augmentation techniques were used:
- Horizontal flipping
- Rotation
- Zoom
- Contrast adjustment

## CNN Architecture
Input Image (128 × 128 × 3)

↓ Convolution Layer (32 filters)

↓ Max Pooling

↓ Convolution Layer (64 filters)

↓ Max Pooling

↓ Convolution Layer (128 filters)

↓ Max Pooling

↓ Flatten

↓ Dense Layer (128 neurons)

↓ Softmax Output Layer

↓ 10 Classes

## Run the Application

Install required libraries:

```bash
pip install -r requirements.txt