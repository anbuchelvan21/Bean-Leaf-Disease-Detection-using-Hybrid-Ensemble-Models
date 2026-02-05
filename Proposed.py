# ==============================================================
# 🌿 HYBRID ENSEMBLE TRAINING & EVALUATION PIPELINE
# Models: EfficientNetV2B3 + ResNet50 + MobileNetV2 + MobileNetV3Large
# Weighted Ensemble + Performance Visualization
# ==============================================================

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetV2B3, ResNet50, MobileNetV2, MobileNetV3Large
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, matthews_corrcoef, cohen_kappa_score
from sklearn.preprocessing import label_binarize
from sklearn.utils import class_weight
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
import os

# ---------------------------
# Global parameters
# ---------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
LEARNING_RATE_STAGE_1 = 3e-5
LEARNING_RATE_STAGE_2 = 1e-6 
DROPOUT_RATE = 0.4
L2_REG = 1e-4
FINE_TUNE_EFFICIENT_RESNET = 40
FINE_TUNE_MOBILENET = 30
EPOCHS_STAGE_1 = 25
EPOCHS_STAGE_2 = 10
PATIENCE = 10

# ---------------------------
# Paths
# ---------------------------
train_dir = r"C:\Users\MSI\Desktop\Course Projects\TARP\train"
val_dir   = r"C:\Users\MSI\Desktop\Course Projects\TARP\validation"
test_dir  = r"C:\Users\MSI\Desktop\Course Projects\TARP\test"
CHECKPOINT_DIR = 'model_checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ---------------------------
# Compute feature-wise mean
# ---------------------------
print("Computing dataset mean for feature-wise centering...")
temp_gen = ImageDataGenerator()
temp_flow = temp_gen.flow_from_directory(train_dir, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode="categorical")
train_images_sample = [temp_flow[i][0] for i in range(min(len(temp_flow), 10))]
train_images_sample = np.vstack(train_images_sample)
featurewise_mean = np.mean(train_images_sample, axis=(0,1,2))
print("Feature-wise mean computed:", featurewise_mean)

# ---------------------------
# Data Augmentation
# ---------------------------
train_gen = ImageDataGenerator(
    featurewise_center=True,
    horizontal_flip=True,
    rotation_range=25,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2],
    channel_shift_range=30.0
)
train_gen.mean = featurewise_mean

val_gen = ImageDataGenerator(featurewise_center=True)
val_gen.mean = featurewise_mean
test_gen = ImageDataGenerator(featurewise_center=True)
test_gen.mean = featurewise_mean

# ---------------------------
# Generators
# ---------------------------
train_flow = train_gen.flow_from_directory(train_dir, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode="categorical")
val_flow = val_gen.flow_from_directory(val_dir, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode="categorical")
test_flow = test_gen.flow_from_directory(test_dir, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False)
num_classes = train_flow.num_classes

# ---------------------------
# Class Weights
# ---------------------------
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(train_flow.classes), y=train_flow.classes)
CLASS_WEIGHTS_DICT = dict(enumerate(class_weights))
print("\nCalculated Class Weights:", CLASS_WEIGHTS_DICT)

# ---------------------------
# Model Builder
# ---------------------------
def build_model(base_model_fn, fine_tune_layers, dense_units, base_model_name):
    if base_model_name == "MobileNetV3Large":
        base = base_model_fn(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE,3), minimalistic=True)
    else:
        base = base_model_fn(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE,3))
    
    for layer in base.layers:
        layer.trainable = False
    for layer in base.layers[-fine_tune_layers:]:
        layer.trainable = True

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(dense_units, activation="relu", kernel_regularizer=l2(L2_REG))(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    out = layers.Dense(num_classes, activation="softmax", kernel_regularizer=l2(L2_REG))(x)
    return models.Model(inputs=base.input, outputs=out)

efficient_model = build_model(EfficientNetV2B3, FINE_TUNE_EFFICIENT_RESNET, 512, "EfficientNetV2B3")
resnet_model = build_model(ResNet50, FINE_TUNE_EFFICIENT_RESNET, 512, "ResNet50")
mobilenet_model = build_model(MobileNetV2, FINE_TUNE_MOBILENET, 256, "MobileNetV2")
mobilenetv3_model = build_model(MobileNetV3Large, FINE_TUNE_MOBILENET, 256, "MobileNetV3Large")

# ---------------------------
# Training Utilities
# ---------------------------
def get_callbacks(model_name):
    filepath = os.path.join(CHECKPOINT_DIR, f"best_weights_{model_name}.h5")
    early_stop = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0, save_best_only=True, mode='max')
    return [early_stop, checkpoint], filepath

def compile_model(model, lr, use_scheduler=False):
    if use_scheduler:
        lr_schedule = CosineDecay(initial_learning_rate=lr, decay_steps=int(train_flow.samples / BATCH_SIZE) * EPOCHS_STAGE_2, alpha=0.1)
        final_lr = lr_schedule
    else:
        final_lr = lr
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=final_lr),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=["accuracy"])

def deep_fine_tune(model, history_initial, name, best_weights_path):
    print(f"\nStarting Deep Fine-Tuning (Stage 2) for {name}...")
    for layer in model.layers:
        layer.trainable = True
    compile_model(model, LEARNING_RATE_STAGE_2, use_scheduler=True)
    early_stop = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    checkpoint = ModelCheckpoint(best_weights_path, monitor='val_accuracy', verbose=0, save_best_only=True, mode='max')
    history_deep = model.fit(train_flow, validation_data=val_flow, epochs=EPOCHS_STAGE_2, callbacks=[early_stop, checkpoint], class_weight=CLASS_WEIGHTS_DICT, verbose=1)
    for key in history_initial.history:
        if key in history_deep.history:
            history_initial.history[key].extend(history_deep.history[key])
    return history_initial

# ---------------------------
# Train Models
# ---------------------------
models_to_train = [efficient_model, resnet_model, mobilenet_model, mobilenetv3_model]
model_names = ["EfficientNetV2B3", "ResNet50", "MobileNetV2", "MobileNetV3Large"]
histories = []

for model, name in zip(models_to_train, model_names):
    callbacks_s1, path_s1 = get_callbacks(name)
    compile_model(model, LEARNING_RATE_STAGE_1)
    print(f"\nTraining {name} (Stage 1)...")
    history_s1 = model.fit(train_flow, validation_data=val_flow, epochs=EPOCHS_STAGE_1, callbacks=callbacks_s1, class_weight=CLASS_WEIGHTS_DICT, verbose=1)
    model.load_weights(path_s1)
    history_combined = deep_fine_tune(model, history_s1, name, path_s1)
    histories.append(history_combined)
    model.load_weights(path_s1)

# ---------------------------
# Weighted Ensemble Prediction
# ---------------------------
val_accs = [max(hist.history['val_accuracy']) for hist in histories]
weights = [acc / sum(val_accs) for acc in val_accs]
print(f"\nEnsemble Weights: {dict(zip(model_names, [f'{w:.4f}' for w in weights]))}")

def ensemble_predict_weighted(models, weights, test_gen):
    probs = [m.predict(test_gen, verbose=0) for m in models]
    weighted_probs = np.average(probs, axis=0, weights=weights)
    y_pred = np.argmax(weighted_probs, axis=1)
    return y_pred, weighted_probs

y_pred, y_pred_probs = ensemble_predict_weighted(models_to_train, weights, test_flow)
y_true = test_flow.classes

# ---------------------------
# Metrics Calculation
# ---------------------------
class_names_list = list(train_flow.class_indices.keys())
class_report_text = classification_report(y_true, y_pred, target_names=class_names_list)
auc = roc_auc_score(tf.keras.utils.to_categorical(y_true, num_classes=num_classes), y_pred_probs, multi_class="ovr")
mcc = matthews_corrcoef(y_true, y_pred)
kappa = cohen_kappa_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print("\nMetrics Summary:")
print(f"AUC: {auc:.4f}")
print(f"MCC: {mcc:.4f}")
print(f"Cohen's Kappa: {kappa:.4f}")
print("\nClassification Report:\n", class_report_text)

# ---------------------------
# 📊 ENSEMBLE EVALUATION (Train / Val / Test)
# ---------------------------
def ensemble_evaluate(models, weights, generator):
    probs = [m.predict(generator, verbose=0) for m in models]
    weighted_probs = np.average(probs, axis=0, weights=weights)
    y_pred = np.argmax(weighted_probs, axis=1)
    y_true = generator.classes
    loss = tf.keras.losses.categorical_crossentropy(
        tf.keras.utils.to_categorical(y_true, num_classes=len(generator.class_indices)),
        weighted_probs
    ).numpy().mean()
    acc = np.mean(y_pred == y_true)
    return loss, acc

train_loss, train_acc = ensemble_evaluate(models_to_train, weights, train_flow)
val_loss, val_acc = ensemble_evaluate(models_to_train, weights, val_flow)
test_loss, test_acc = ensemble_evaluate(models_to_train, weights, test_flow)

print("\n🔹 Ensemble Performance Summary:")
print(f"Train →  Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}")
print(f"Val   →  Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}")
print(f"Test  →  Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}")

# ---------------------------
# 📈 Smooth Weighted Ensemble Learning Curve
# ---------------------------
def plot_ensemble_learning_curve(histories, model_names, weights):
    all_epochs = range(max(len(h.history['accuracy']) for h in histories))
    weighted_train_acc = np.zeros(len(all_epochs))
    weighted_val_acc = np.zeros(len(all_epochs))

    for hist, w in zip(histories, weights):
        epochs = len(hist.history['accuracy'])
        weighted_train_acc[:epochs] += np.array(hist.history['accuracy']) * w
        weighted_val_acc[:epochs] += np.array(hist.history['val_accuracy']) * w

    plt.figure(figsize=(8,5))
    plt.plot(weighted_train_acc, label="Train (Weighted Avg)", color="#2E86C1", linewidth=2.5)
    plt.plot(weighted_val_acc, '--', label="Validation (Weighted Avg)", color="#E67E22", linewidth=2.5)
    plt.title("Smooth Ensemble Learning Curve", fontsize=14, fontweight='bold')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.show()

# ---------------------------
# 📊 Plot Ensemble Summary (Bar Chart)
# ---------------------------
plt.figure(figsize=(9,5))
datasets = ['Train', 'Validation', 'Test']
acc_values = [train_acc, val_acc, test_acc]
loss_values = [train_loss, val_loss, test_loss]

plt.subplot(1,2,1)
plt.bar(datasets, acc_values, color=['#2E86C1','#48C9B0','#F4D03F'], edgecolor='black')
plt.title("Ensemble Accuracy", fontsize=13, fontweight='bold')
plt.ylim(0,1)
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.subplot(1,2,2)
plt.bar(datasets, loss_values, color=['#2E86C1','#48C9B0','#F4D03F'], edgecolor='black')
plt.title("Ensemble Loss", fontsize=13, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.suptitle("Weighted Ensemble Performance Summary", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()

# Call the smooth learning curve plot
plot_ensemble_learning_curve(histories, model_names, weights)
