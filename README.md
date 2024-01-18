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

## Approach 3: Convolutional Neural Network (CNN) for CAPTCHA Letter Recognition
### 1. Introduction
The primary goal of Approach 3 is to create a robust solution for recognizing
individual letters within CAPTCHA images using Convolutional Neural Networks (CNN). CAPTCHAs, commonly employed for user verification on websites, present a challenge due to the presence of noise, such as blurring and lines,
making traditional image recognition approaches less effective. This report outlines the key steps and methodologies employed in building and training a CNN
model for letter recognition.
### 2. Image Processing
#### 2.1 Overview
The initial step involves preprocessing the CAPTCHA images to enhance their
suitability for CNN-based recognition. Image processing techniques are employed to handle noise, adaptively threshold the images, and partition them
into distinct regions corresponding to individual letters.
#### 2.2 Adaptive Thresholding
The application of adaptive thresholding, using the Gaussian method, is crucial
in addressing variations in illumination across CAPTCHA images. This method
ensures effective binarization, enhancing the model’s ability to distinguish letter
shapes.
#### 2.3 Closing and Dilation
Closing operations are employed to address gaps in between object regions,
ensuring a more coherent representation of letters. Dilation further aids in
expanding the white regions in the image, refining the letter boundaries.
#### 2.4 Smoothing Images (Blurring)
Gaussian blurring is utilized to remove high-frequency components, such as noise
and edges, contributing to cleaner and more discernible letter shapes.
#### 2.5 Partitioning
The process of partitioning involves segmenting the image into five regions,
each representing an individual letter in the CAPTCHA. This step is pivotal in
preparing distinct inputs for the subsequent CNN model.

### 3. Initial Analysis and Data Wrangling
#### 3.1 Scaling
Normalization is performed to scale pixel values between 0 and 1, creating a
standardized input for the CNN.
#### 3.2 Label Distribution Analysis
An analysis of label distribution reveals insights into the dataset. This understanding aids in addressing potential biases and optimizing the model’s performance.
#### 3.3 One-Hot Encoding
Labels are one-hot encoded to represent categorical information, allowing for
effective training of the CNN.
#### 3.4 Train-Test Split
The dataset is divided into training and testing sets to facilitate model evaluation. This ensures the model’s ability to generalize to unseen data.
### 4. Model Creation
A CNN model is designed, consisting of convolutional and dense layers. This
architecture is tailored to capture hierarchical features within the images, crucial
for distinguishing different letter shapes.
### 5. Data Augmentation and Oversampling
#### 5.1 SMOTE
Oversampling is performed using SMOTE to balance the dataset and prevent
biases toward specific letters. This technique enhances the model’s ability to
generalize to underrepresented classes.
#### 5.2 ImageDataGenerator
Data augmentation, incorporating rotations and shifts, is applied to diversify
the training dataset further. This step aids in improving the model’s robustness
and adaptability to variations in CAPTCHA presentations.
### 6. Model Training
The CNN model is trained using the prepared dataset. ModelCheckpoint and
ReduceLROnPlateau callbacks are implemented to ensure efficient training and
adaptability to changing data patterns.
### 7. Model Evaluation
#### 7.1 Model Performance Visualization
Graphical representations of loss and accuracy throughout the training process
offer insights into the model’s convergence and overall performance.

#### 7.2 Model Prediction and Evaluation
The trained model is evaluated on the test set, and accuracy metrics are calculated. Classification reports provide a detailed breakdown of performance across
different letters

## Approach 4: Enhanced CRNN Model for CAPTCHA Letter Recognition
### 1. Introduction
The primary goal of Approach 4 is to refine the CRNN (Convolutional Recurrent Neural Network) model for effective CAPTCHA letter recognition. This
approach incorporates improvements in the model architecture and training process to enhance overall performance. While the accuracy may not be perfect,
it demonstrates significant progress in predicting a substantial portion of the
captcha, making it a valuable addition to the evaluation process.
### 2. Model Enhancements
#### 2.1 Convolutional Layer Adjustment
The CRNN model underwent adjustments in convolutional layer configurations
to adapt better to CAPTCHA image characteristics. These modifications aim
to capture more intricate features crucial for recognizing individual letters.
#### 2.2 Increased Epochs
Approach 4 was trained for a modest 10 epochs, allowing the model to learn
intricate patterns within the limited timeframe. Further training may lead to
improved accuracy.
### 3. Training Process
#### 3.1 Optimizer Update
Stochastic Gradient Descent (SGD) with Nesterov momentum was employed as
the optimizer. The learning rate, weight decay, and momentum were fine-tuned
for optimal training performance.
##### 3.2 Training Dynamics
The model was trained on a dataset with a varied representation of CAPTCHA
images. The training process included monitoring training and validation losses
for effective model convergence.
### 4. Evaluation Results
Approach 4 demonstrated notable progress in predicting CAPTCHA letters accurately. While the exact accuracy might not be optimal, the model consistently
predicts a significant portion of the captcha, showcasing its potential for further
improvement. It achieved an accuracy of 47.78% on training dataset and an
accuracy of 47.37% on test dataset and 77.73% on validation dataset. All the
captcha letters were converted into small letters and then calculated.
### 5. Sample Predictions
A sample of 10 predictions from Approach 4 illustrates the model’s ability to
approximate CAPTCHA solutions. Despite the occasional deviation from the
actual captcha, the predictions exhibit promising outcomes, especially considering the limited training duration.
### 6. Comparative Analysis
Approach 4 provides an alternative perspective on CAPTCHA recognition,
achieving commendable results in predicting a substantial portion of the captcha.
While the exact accuracy might fall short, the model’s ability to smoothly predict 3/4 letters marks a significant advancement.
### 7. Conclusion and Future Directions
Approach 4 contributes valuable insights into enhancing CAPTCHA recognition models. While further training and optimization are recommended, the
progress achieved in predicting a significant portion of the captcha positions
this approach as a promising candidate for future iterations.
### 8. Recommendations
• 1. Extend Training: Consider training Approach 4 for an increased number of epochs to allow the model to further learn intricate patterns.
• 2. Fine-Tuning: Experiment with hyperparameter tuning and model architecture adjustments to enhance accuracy.
• 3. Iterative Development: Emphasize continuous refinement and exploration of advanced techniques to address real-world challenges.

## Comparative Analysis and Insights
### Accuracy Comparison
#### Approach 1: Exhibited superior accuracy on the test dataset.
#### Approach 2: Limited success with significantly lower accuracy compared to Approach 1.
#### Approach 3: Faced challenges with poor image quality, resulting in bad accuracy.
#### Approach 4: Demonstrated progress in predicting a substantial portion of the captcha, especially considering its brief training duration.

### Scalability and Robustness
#### Approach 1:
1. Inherently scalable due to the flexibility of neural network architectures.
2. Demonstrated robustness through training on a diverse dataset.
#### Approach 2:
1. Contour-based approach may face challenges with scalability.
2. Struggled with variations in captcha design and letter complexities.
#### Approach 3:
1. Encountered difficulties in scalability and robustness due to image quality
issues.
#### Approach 4:
1. Despite limited training epochs, showcased potential in predicting a significant portion of the captcha.

## Conclusion and Future Directions
###Preferred Approach
#### Approach 1:
1. Emerges as the more robust and accurate solution.
2. Demonstrated superior accuracy and adaptability.
