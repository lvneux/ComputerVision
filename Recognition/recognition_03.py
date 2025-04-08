import tensorflow as tf
from tensorflow import keras
from keras.applications import VGG16
from keras import layers, models
from keras.datasets import cifar10
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0 

base_model = VGG16(weights='imagenet', 
                   include_top=False, 
                   input_shape=(32, 32, 3))  

for layer in base_model.layers:
    layer.trainable = False 

model = models.Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')  
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, 
                    epochs=10,
                    validation_split=0.2)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'\nVGG16 테스트 정확도: {test_acc:.4f}')

baseline_model = models.Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

baseline_model.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

baseline_history = baseline_model.fit(x_train, y_train, 
                                      epochs=10,
                                      validation_split=0.2)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'\nCNN 테스트 정확도: {test_acc:.4f}')

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['val_accuracy'], label='VGG16 Transfer')
plt.plot(baseline_history.history['val_accuracy'], label='Baseline CNN')
plt.title('Validation Accuracy Comparison')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['val_loss'], label='VGG16 Transfer')
plt.plot(baseline_history.history['val_loss'], label='Baseline CNN')
plt.title('Validation Loss Comparison')
plt.legend()
plt.show()
