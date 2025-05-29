# -----------------------------
# 1. Import Libraries
# -----------------------------
# For numerical operations (e.g., working with image arrays)
import numpy as np
# To measure performance
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt  # For drawing graphs (like accuracy over time)

# Keras modules for building CNN models
from tensorflow.keras.models import Sequential  # To build a layer-by-layer model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
# - Conv2D: Learns patterns in images (like eyes, mouths)
# - MaxPooling2D: Zooms out to see bigger patterns
# - Flatten: Turns image into a list of numbers
# - Dense: Decides which emotion it is
# - Dropout: Helps avoid cheating (overfitting)
# - BatchNormalization: Stabilizes learning

# Prepares images for training
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Smart optimizer that helps model learn faster
from tensorflow.keras.optimizers import Adam
# Tools to improve training
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# -----------------------------
# 2. Define Paths and Parameters
# -----------------------------
train_dir = 'dataset/train'  # Folder where training images are stored
test_dir = 'dataset/test'     # Folder where test images are stored

# All images will be resized to 48x48 pixels (standard for FER2013)
img_size = 48
batch_size = 64    # How many images to process at once (64 is a good balance)
epochs = 30        # How many times to go through all data during training
# Number of emotions (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise)
num_classes = 7


# -----------------------------
# 3. Data Augmentation & Generators
# -----------------------------
# Prepare training images with random changes to help model generalize better
datagen_train = ImageDataGenerator(
    rescale=1./255,         # Normalize pixel values from [0,255] to [0,1]
    rotation_range=15,       # Slightly rotate images
    width_shift_range=0.1,   # Move images left/right a little
    height_shift_range=0.1,  # Move images up/down a little
    shear_range=0.1,         # Skew images slightly
    zoom_range=0.2,          # Zoom in/out slightly
    horizontal_flip=True     # Flip some images left-right
)

# Test images only get normalized (no augmentation)
datagen_test = ImageDataGenerator(rescale=1./255)


# -----------------------------
# 4. Load Images from Folders
# -----------------------------
# Prepare training data with augmentation
train_generator = datagen_train.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),  # Resize all images to 48x48
    color_mode='grayscale',           # Use black-and-white images
    batch_size=batch_size,
    class_mode='categorical',         # Tells model there are multiple classes
    shuffle=True                      # Mix up data during training
)

# Prepare testing data without mixing up order
test_generator = datagen_test.flow_from_directory(
    test_dir,
    target_size=(img_size, img_size),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False                     # Keep test data in order for evaluation
)


# -----------------------------
# 5. Build the CNN Model
# -----------------------------
# Create a brain-like structure to recognize emotions
model = Sequential([
    # First Layer: Learn basic patterns
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    BatchNormalization(),     # Stabilizes learning
    MaxPooling2D((2, 2)),   # Zooms out to see bigger patterns

    # Second Layer: Learn more complex patterns
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    # Third Layer: Learn advanced features like facial expressions
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    # Turn image into a list of numbers
    Flatten(),

    # Make decisions based on what it learned
    Dense(256, activation='relu'),  # 256 neurons decide what’s important
    Dropout(0.5),                   # Helps avoid overfitting (cheating)
    Dense(7, activation='softmax')  # Final decision: pick one of 7 emotions
])


# -----------------------------
# 6. Compile the Model
# -----------------------------
# Set up how the model learns
model.compile(optimizer=Adam(learning_rate=0.001),  # Controls learning speed
              loss='categorical_crossentropy',      # Measures how wrong predictions are
              # Track how often it's right
              metrics=['accuracy'])


# -----------------------------
# 7. Train the Model with Callbacks
# -----------------------------
# Stop early if model stops improving
early_stop = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True)

# If stuck, slow down learning rate
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# Start training the model
history = model.fit(
    train_generator,            # Training images
    epochs=epochs,             # How long to train
    validation_data=test_generator,  # Test images to check progress
    callbacks=[early_stop, reduce_lr]  # Tools to make learning smarter
)


# -----------------------------
# 8. Evaluate on Test Set
# -----------------------------
# Test how well the model works on new images
test_loss, test_acc = model.evaluate(test_generator)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")


# -----------------------------
# 9. Plot Training History
# -----------------------------
# Draw graphs to show how model improved
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Also show how much the model was "wrong" (lower is better)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# -----------------------------
# 10. Predict Classes and Show Confusion Matrix
# -----------------------------
# Reset generator to start from beginning
test_generator.reset()

# Ask model to guess every emotion in test set
preds = model.predict(test_generator)

# Get predicted labels (which emotion it guessed)
y_pred = np.argmax(preds, axis=1)

# Get real labels (what the correct answers are)
y_true = test_generator.classes

# List of emotion names
class_names = list(test_generator.class_indices.keys())

# Show how well the model did for each emotion
cm = confusion_matrix(y_true, y_pred)
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))


# -----------------------------
# 11. Show Some Example Predictions
# -----------------------------
# Get one batch of test images and their true labels
x_test, y_test = next(test_generator)

# Ask model to predict these images
predictions = model.predict(x_test)

# Show 5 example images with predictions vs truth
for i in range(5):
    plt.imshow(x_test[i].reshape(48, 48), cmap='gray')  # Show image
    pred_label = class_names[np.argmax(predictions[i])]  # What model guessed
    true_label = class_names[np.argmax(y_test[i])]        # What was correct
    plt.title(f"Predicted: {pred_label}\nTrue: {true_label}")
    plt.axis('off')
    plt.show()
