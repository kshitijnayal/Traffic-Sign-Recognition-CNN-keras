# Traffic Signs Recognition using CNN

This project is a Convolutional Neural Network (CNN)-based Traffic Sign Recognition system built using Python and TensorFlow/Keras. It classifies traffic signs into 43 categories using image data and deep learning techniques.

## Features
- Preprocessing and augmentation of traffic sign images
- Model built using Conv2D, MaxPooling, and Dropout layers
- Trained on 39,000+ images with high accuracy (~99% val accuracy)
- Plots training/validation accuracy and saves the model (`.keras` format)

## Technologies Used
- Python
- TensorFlow / Keras
- NumPy, Pandas, Matplotlib
- Scikit-learn for train/test split and evaluation

## Dataset
Used the [German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/) dataset for training and testing.

## Results
- Achieved >98% accuracy on validation set
- Model saved for future inference
