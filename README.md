# CIFAR-10 Image Classification using CNN

This project implements a **Convolutional Neural Network (CNN)** to classify images from the **CIFAR-10 dataset**. The dataset consists of 60,000 32x32 color images categorized into 10 classes, such as airplanes, cars, birds, cats, etc. This project demonstrates the use of CNNs for image classification, data augmentation, and model evaluation using TensorFlow and Keras.

## Dataset

The **CIFAR-10 dataset** contains 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The dataset is split into 50,000 training images and 10,000 test images. The classes in the dataset are as follows:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

The dataset is already available in the TensorFlow dataset library and can be loaded directly.

## Model Architecture

The CNN used for this project consists of the following layers:
1. **Input Layer**: Input shape of 32x32x3 (corresponding to CIFAR-10 images).
2. **Convolutional Layers**: Multiple Conv2D layers with ReLU activation and Batch Normalization to extract features from the images.
3. **Pooling Layers**: MaxPooling2D layers to downsample the feature maps.
4. **Flattening Layer**: Converts the 2D feature maps into a 1D feature vector.
5. **Fully Connected Layers**: Dense layers with ReLU activation to perform classification.
6. **Output Layer**: Dense layer with softmax activation to predict one of the 10 classes.

### Code Overview:

```python
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Input, BatchNormalization, Flatten
from tensorflow.keras.models import Model

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Model architecture
i = Input(shape=x_train[0].shape)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(i)
x = BatchNormalization()(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2, 2)(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2, 2)(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2, 2)(x)

# Fully connected layers
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(10, activation='softmax')(x)

# Model creation
model = Model(i, x)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model on the dataset
model.fit(x_train, y_train, epochs=50)

# Data augmentation using ImageDataGenerator
batch_size = 32
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range= 0.1, height_shift_range=0.1, rotation_range=0.2)
train_generator = data_generator.flow(x_train, y_train, batch_size)
steps_per_epoch = x_train.shape[0] // batch_size
model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=50)

# Evaluate the model on the test set
y_pred = model.predict(x_test).argmax(axis=1)

# Calculate accuracy and confusion matrix
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy = accuracy_score(y_pred=y_pred, y_true=y_test)

```

## Results

After training the model for **50 epochs** with additional **data augmentation** (image shifts and rotations), the model was evaluated on the CIFAR-10 test set. Below are the key results and metrics:

- **Test Accuracy**: `~0.8592`

