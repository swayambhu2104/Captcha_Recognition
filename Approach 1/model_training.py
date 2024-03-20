import tensorflow as tf
from pathlib import Path
import pandas as pd
import os
import numpy as np
from data_preprocessing import encode_single_sample
from data_preprocessing import split_data
from data_preprocessing import build_model
import time
import tensroflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from data_preprocessing import decode_batch_predictions
from tensorflow.keras import layers


batch_size = 16

# Path to the data directory
data_dir = Path("data")

# Get list of all the images
images = sorted(list(map(str, list(data_dir.glob("*.jpg")))))
labels = [img.split(os.path.sep)[-1].split(".jpg")[0] for img in images]
characters = set(char for label in labels for char in label)
characters = sorted(list(characters))

print("Number of images found: ", len(images))
print("Number of labels found: ", len(labels))
print("Number of unique characters: ", len(characters))
print("Characters present: ", characters)

# Maximum length of any captcha in the dataset
max_length = max([len(label) for label in labels])

# Mapping characters to integers
char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)

# Mapping integers back to original characters
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

x_train, x_valid, x_test, y_train, y_valid, y_test = split_data(np.array(images), np.array(labels), train_size=0.6, validation_size=0.2)



train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = (
    train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
validation_dataset = (
    validation_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = (
    test_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

model = build_model()
# TODO restore epoch count.
start_time = time.time()
epochs = 100
early_stopping_patience = 6
model_check_point = tf.keras.callbacks.ModelCheckpoint("/content/drive/MyDrive/model_01",monitor="val_loss",save_best_only=True)
# Add early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    callbacks=[early_stopping,
               model_check_point],
)
end_time = time.time()
elapsed_time = (end_time - start_time)/60
print(f"Time taken: {elapsed_time:.2f} minutes")

model.save('captcha_model')


prediction_model = keras.models.Model(
    model.input[0], model.get_layer(name="dense2").output
)

#  Let's check results on some validation samples
for batch in validation_dataset.take(1):
    batch_images = batch["image"]
    batch_labels = batch["label"]

    preds = prediction_model.predict(batch_images)
    pred_texts = decode_batch_predictions(preds)

    orig_texts = []
    for label in batch_labels:
        label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
        orig_texts.append(label)

    _, ax = plt.subplots(4, 4, figsize=(15, 5))
    for i in range(len(pred_texts)):
        img = (batch_images[i, :, :, 0] * 255).numpy().astype(np.uint8)
        img = img.T
        title = f"Prediction: {pred_texts[i]}"
        ax[i // 4, i % 4].imshow(img, cmap="gray")
        ax[i // 4, i % 4].set_title(title)
        ax[i // 4, i % 4].axis("off")
plt.show()


test_acc = calculate_accuracy(test_dataset,"test_dataset")
val_accuracy = calculate_accuracy(validation_dataset,"val_dataset")
train_acc = calculate_accuracy(train_dataset,"train_dataset")     
data = {'Dataset': ['Training', 'Validation', 'Test'],
        'Accuracy': [train_acc, val_accuracy, test_acc]}

accuracy_df = pd.DataFrame(data)
print(accuracy_df)