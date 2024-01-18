# Captcha Recognition
## Approach 1: Neural Network with CTC Loss (Final Model)
### Data Preprocessing and Exploration
#### 1.1 Dataset Overview:
1. The initial dataset consisted of 113,062 captcha images in JPG format.
2. Images were associated with corresponding labels derived from filenames.
#### 1.2 Data Splitting:
1. The team employed a standard 60-20-20 split for training, testing, and
validation sets.
2. Resulted in 67,837 training images, 22,612 testing images, and 22,613
validation images.
#### 1.3 Image Quality Enhancement:
1. Recognizing suboptimal image quality, the team converted all images to
grayscale.
2. OpenCV was utilized to address blurriness, ensuring cleaner inputs for the
OCR model.

## Model Architecture
### OCR Model Architecture 
The OCR model architecture is a crucial component of the approach. The model is structured as an encoder-decoder, incorporating convolutional layers for image feature extraction, dense layers for mapping, and bidirectional LSTM layers for sequence modeling. The CTC loss
layer is added to facilitate effective training.
#### 2.1 Convolutional Blocks:
1. Initial block: 32 filters, 3x3 kernel, ReLU activation, and max-pooling.
2. Second block: 64 filters with a similar structure.
#### 2.2 Reshape Layer:
Data reshaped to prepare for subsequent Recurrent Neural Network (RNN) layers.
#### 2.3 Dense Layer:
A dense layer with 64 units, ReLU activation, and dropout for regularization.
#### 2.4 Recurrent Layers:
Two Bidirectional Long Short-Term Memory (LSTM) layers with dropout for capturing temporal dependencies.
#### 2.5 Output Layer:
Dense layer with softmax activation for character prediction.
#### 2.6 CTC Loss Layer:
Custom layer (‘CTCLayer‘) integrated to compute the Connectionist Temporal Classification (CTC) loss.
