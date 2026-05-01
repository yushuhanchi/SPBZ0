import os
import numpy as np
import pandas as pd
import librosa

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf  
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import (  
    Input,
    Conv2D,
    MaxPooling2D,
    Dropout,
    Flatten,
    Dense,
)
from tensorflow.keras.utils import to_categorical  
DATASET_PATH = r"C:\Users\ASUS\Desktop\SPBZ0\UrbanSound8K"

metadata_path = os.path.join(DATASET_PATH, "metadata", "UrbanSound8K.csv")
audio_path = os.path.join(DATASET_PATH, "audio")

print("Metadata exists:", os.path.exists(metadata_path))
print("Audio folder exists:", os.path.exists(audio_path))

metadata = pd.read_csv(metadata_path)

print(metadata.head())
print(metadata["class"].value_counts())

# 2. Extract MFCC features
def extract_mfcc(file_path, max_pad_len=174):
    try:
        audio, sample_rate = librosa.load(file_path, sr=22050)
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(
                mfcc,
                pad_width=((0, 0), (0, pad_width)),
                mode="constant"
            )
        else:
            mfcc = mfcc[:, :max_pad_len]

        return mfcc

    except Exception as e:
        print("Error:", file_path, e)
        return None


features = []
labels = []

print("Extracting MFCC features...")

for index, row in metadata.iterrows():
    file_name = row["slice_file_name"]
    fold = "fold" + str(row["fold"])
    label = row["class"]

    file_path = os.path.join(audio_path, fold, file_name)

    if not os.path.exists(file_path):
        print("File not found:", file_path)
        continue

    mfcc = extract_mfcc(file_path)

    if mfcc is not None:
        features.append(mfcc)
        labels.append(label)

    if index % 500 == 0:
        print(f"Processed {index} files")

X = np.array(features)
y = np.array(labels)

print("Feature shape before standardisation:", X.shape)
print("Labels shape:", y.shape)

# 3. Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# 4. Train-test split first
X_train, X_test, y_train, y_test, y_train_encoded, y_test_encoded = train_test_split(
    X,
    y_categorical,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# 5. Standardisation
# Use training data mean/std only to avoid data leakage
mean = np.mean(X_train)
std = np.std(X_train) + 1e-8

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# 6. Reshape for Conv2D
# From (samples, 40, 174) to (samples, 40, 174, 1)
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# 7. Build improved Conv2D model
model = Sequential([
    Input(shape=(40, 174, 1)),

    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Flatten(),

    Dense(128, activation="relu"),
    Dropout(0.4),

    Dense(10, activation="softmax")
])

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()

# 8. Train model
history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2
)

# 9. Evaluate model
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)

# 10. Classification report
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print(classification_report(
    y_true_classes,
    y_pred_classes,
    target_names=label_encoder.classes_
))

print("Confusion matrix:")
print(confusion_matrix(y_true_classes, y_pred_classes))

# 11. Save model
model.save("urban_noise_classifier_improved.keras")

print("Model saved as urban_noise_classifier_improved.keras")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 1. 计算 confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)

# 2. 画图
plt.figure(figsize=(10,8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks(rotation=45)
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

plt.figure(figsize=(8,6))

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title("Model Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.grid(True)
plt.savefig("training_curve.png", dpi=300)
plt.show()