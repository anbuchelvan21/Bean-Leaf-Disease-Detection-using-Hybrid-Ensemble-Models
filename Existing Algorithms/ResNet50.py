import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50, resnet50
from tensorflow.keras import layers, models, optimizers
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np

# Dataset Paths
train_dir = r"C:\Users\MSI\Desktop\TARP\train"
val_dir   = r"C:\Users\MSI\Desktop\TARP\validation"
test_dir  = r"C:\Users\MSI\Desktop\TARP\test"

IMG_SIZE = 224
BATCH_SIZE = 32

# Load Dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir, image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, label_mode="categorical"
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir, image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, label_mode="categorical"
)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir, image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, label_mode="categorical", shuffle=False
)

num_classes = len(train_ds.class_names)
print("Classes:", train_ds.class_names)

# Preprocessing for ResNet50
def preprocess(image, label):
    image = resnet50.preprocess_input(image)
    return image, label

train_ds = train_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)
test_ds  = test_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)

# Build Model
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
for layer in base_model.layers[:-30]:  
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax")
])

# Compile & Train
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=[early_stop]
)

# Evaluate
loss, acc = model.evaluate(test_ds)
print(f"\nFinal Test Accuracy: {acc*100:.2f}%")

# Plot Accuracy & Loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.legend()
plt.title("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Loss")
plt.show()

# Confusion Matrix
y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_true = np.argmax(y_true, axis=1)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_ds.class_names)

plt.figure(figsize=(8,6))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix")
plt.show()

# Classification Report
report = classification_report(y_true, y_pred, target_names=train_ds.class_names, digits=4)
print("Classification Report (Precision, Recall, F1-score per class):\n")
print(report)
