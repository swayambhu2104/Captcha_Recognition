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

### Model Architecture
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

### Training Strategy
#### 3.1 Epochs and Early Stopping:
Model trained for 100 epochs. Early stopping implemented to prevent overfitting.
#### 3.2 Optimizer:
Adam optimizer chosen for its efficiency in training neural networks.
### Model Evaluation

#### 4.1 Accuracy Calculation:
A dedicated function (calculate accuracy(dataset)) processed the test dataset. Predictions on batches compared with ground truth labels. Overall accuracy calculated on the test dataset.

#### 4.2 Accuracy Results:
1. **Training Accuracy: 66.16006014416911%**
2. **Validation Accuracy: 60.84379975234389%**
3. **Testing Accuracy: 60.314863131826826%**

## Approach 2: Contour-Based Letter Extraction and Classification
### Image Processing and Letter Extraction
1. Individual Letter Extraction:
Image paths of solved captchas collected. Contour detection used for letter extraction. Thresholding and contour analysis applied to identify letter regions.

2. Padding for Improved Isolation:
Additional padding introduced around letters for better isolation.

### Classification Model
#### 2.1 CNN Architecture:
Simple CNN architecture employed. Two Convolutional layers, MaxPooling,
Flatten, Dense, and Dropout layers.

#### 2.2 Training Process:
Model trained on individual letter images obtained from captcha solutions.
Early stopping implemented to prevent overfitting during training.

### Evaluation on Unsolved Captchas

#### 3.1 Grayscale Conversion and Thresholding:
Unsolved captcha image loaded and converted to grayscale. Additional padding applied for enhanced processing.

#### 3.2 Contour-Based Letter Extraction:
Contours identified to extract individual letters. Handling of wide contours to address potential letter mergers.

#### 3.3 Predictions and Annotation:
Predictions made on individual letters using the trained CNN model. Predicted
letters combined to form the captcha solution. Annotated image displayed with predicted letters.

### Observations and Challenges
#### 4.1 Variations and Challenges:
1. Approach 2 faced challenges with variations in letter sizes, font styles, and
inter-letter spacing.
2. The model struggled to distinguish characters in complex captcha images.

#### 4.2 Accuracy Analysis:
1. Approach 2 demonstrated limited success, yielding poor accuracy.
2. Contour-based methods faced difficulties in handling real-world captcha
variations.
