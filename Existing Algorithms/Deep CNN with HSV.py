import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# HSV-based leaf segmentation

def segment_leaf_hsv(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # These ranges may be tuned per your dataset
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    segmented = cv2.bitwise_and(image, image, mask=mask)
    return segmented


# Image preprocessing

def preprocess_image(image_path, img_size=(128, 128)):
    image = cv2.imread(image_path)
    image = segment_leaf_hsv(image)
    image = cv2.resize(image, img_size)
    image = image / 255.0  # normalize
    return image


# Dataset loading

def load_dataset(folder_path, img_size=(128, 128)):
    X = []
    y = []
    class_names = sorted(os.listdir(folder_path))
    for idx, class_name in enumerate(class_names):
        class_folder = os.path.join(folder_path, class_name)
        for file in os.listdir(class_folder):
            file_path = os.path.join(class_folder, file)
            img = preprocess_image(file_path, img_size)
            X.append(img)
            y.append(idx)
    X = np.array(X)
    y = np.array(y)
    return X, y, class_names


# DCNN Model (as per paper)

def build_dcnn(input_shape=(128, 128, 3), num_classes=3):
    model = Sequential()

    # 1st Conv Block
    model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    # 2nd Conv Block
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    # 3rd Conv Block
    model.add(Conv2D(128, (1,1), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    # 4th Conv Block
    model.add(Conv2D(128, (2,2), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    # Flatten & Dense
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model


# Load data

train_path = r"C:\Users\MSI\Desktop\TARP\train"
val_path = r"C:\Users\MSI\Desktop\TARP\validation"
test_path = r"C:\Users\MSI\Desktop\TARP\test"

X_train, y_train, classes = load_dataset(train_path)
X_val, y_val, _ = load_dataset(val_path)
X_test, y_test, _ = load_dataset(test_path)


# Data augmentation

datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)
datagen.fit(X_train)


# Train model

model = build_dcnn(input_shape=(128,128,3), num_classes=len(classes))

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=50,
    verbose=1
)


# Evaluate model

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc*100:.2f}%")


# Plot training history

def plot_history(history):
    plt.figure(figsize=(12,5))
    
    # Accuracy plot
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss plot
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()

plot_history(history)


# Confusion Matrix

def plot_confusion_matrix(model, X_test, y_test, class_names):
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    plt.figure(figsize=(8,6))
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix")
    plt.show()

plot_confusion_matrix(model, X_test, y_test, classes)
