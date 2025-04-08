# 01. 간단한 이미지 분류기 구현

## 과제 설명 및 요구사항
  + 설명
     + 손글씨 숫자 이미지(MNIST 데이터셋)를 이용하여 간단한 이미지 분류기 구현
   
  + 요구사항
      + tensorflow.keras.datasets에서 MNIST 데이터셋 로드 
      + 데이터를 훈련 세트와 테스트 세트로 분할
      + Sequential 모델과 Dense 레이어를 활용하여 간단한 신경망 모델 구축
      + cv.drawKeypoints()를 사용하여 특징점을 이미지에 시각화
      + 모델 훈련 및 정확도 평가
        
## 전체 코드 
   ```
import tensorflow
from tensorflow import keras
from keras.datasets import mnist
from keras import Sequential
from keras.layers import Dense, Flatten

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = Sequential([
    Flatten(input_shape=(28, 28)), 
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    epochs=5,
                    batch_size=32)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'\정확도: {test_acc:.4f}')

 ```

## 데이터 로드 및 전처리 
 ```
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
 ```
+ mnist.load_data()를 사용해 데이터셋 로드 
+ 훈련 데이터 60,000개 & 테스트 데이터 10,000개 자동 분할
+ 0~255 픽셀 값을 0~1 범위로 정규화(Min-Max Sacling)
  
## 신경망 모델 구축 
 ```
model = Sequential([
    Flatten(input_shape=(28, 28)), 
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
 ```
+ Sequential 모델 
+ Flatten: 2D 이미지(28x28) → 1D 벡터(784) 변환
+ Dense(128): 128개 노드의 완전연결층 + ReLU 활성화
+ Dense(10): 10개 클래스 분류를 위한 출력층 + Softmax 활성화

## 모델 컴파일 
 ```
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
 ```
+ optimizer : Adam(Adaptive Moment Estimation) 사용
+ loss function : cross-entropy 사용 (다중 클래스 분류에 적합함)
+ metrics : accuracy(정확도) 사용 

## 모델 훈련
 ```
history = model.fit(x_train, y_train,
                    epochs=5,
                    batch_size=32)
 ```
+ epoch : 전체 데이터셋 5회 반복 학습
+ batch size : 32개 샘플 단위 가중치 갱신
  
## 성능 평가
 ```
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'정확도: {test_acc:.4f}')
 ```
+  model.evaluate()를 사용해 모델 성능 평가
  
## 실행 결과 
+ 정확도: 0.9788


# 02. CIFAR-10 데이터셋을 활용한 CNN 모델 구축

## 과제 설명 및 요구사항
  + 설명
     + CIFAR-10 데이터셋을 활용하여 합성곱 신경망(CNN)을 구축하고, 이미지 분류 수행
   
  + 요구사항
      + tensorflow.keras.datasets에서 CIFAR-10 데이터셋 로드
      + 데이터 전처리(정규화 등) 수행
      + Conv2D, MaxPooling2D, Flatten, Dense 레이어를 활용하여 CNN 모델을 설계하고 훈련
      + 모델 성능 평가 및 테스트 이미지에 대한 예측 수행
        
## 전체 코드 
   ```
import tensorflow as tf
from tensorflow import keras
from keras.datasets import cifar10
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, 
                    epochs=10,
                    validation_split=0.2)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'\n테스트 정확도: {test_acc:.4f}')

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_test[i])
    prediction = tf.argmax(model.predict(x_test[i][tf.newaxis,...])[0])
    true_label = y_test[i][0]
    color = 'blue' if prediction == true_label else 'red'
    plt.xlabel(f"{class_names[prediction]} ({class_names[true_label]})", color=color)
plt.show()

 ```

## 데이터 로드 및 전처리 
 ```
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  
 ```
+ cifar10.load_data()를 사용해 데이터셋 로드 
+ 0~255 픽셀 값을 0~1 범위로 정규화(Min-Max Sacling)
  
## CNN 모델 구축 
 ```
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
 ```
+ Conv2D, MaxPooling2D, Flatten, Dense 레이어를 활용하여 CNN 구성

## 모델 컴파일 
 ```
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
 ```
+ optimizer : Adam(Adaptive Moment Estimation) 사용
+ loss function : cross-entropy 사용 (다중 클래스 분류에 적합함)
+ metrics : accuracy(정확도) 사용 

## 모델 훈련
 ```
history = model.fit(x_train, y_train, 
                    epochs=10,
                    validation_split=0.2)
 ```
+ validation_split을 사용해 validation data 활용 
  
## 성능 평가
 ```
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'정확도: {test_acc:.4f}')
 ```
+  model.evaluate()를 사용해 모델 성능 평가

## 테스트 이미지 예측 시각화 
 ```
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_test[i])
    prediction = tf.argmax(model.predict(x_test[i][tf.newaxis,...])[0])
    true_label = y_test[i][0]
    color = 'blue' if prediction == true_label else 'red'
    plt.xlabel(f"{class_names[prediction]} ({class_names[true_label]})", color=color)
plt.show()
 ```

## 실행 결과 
<img src="https://github.com/user-attachments/assets/07426fa9-81e4-408d-83b8-35dbc31f15e9"/>

+ 정확도 : 0.6943
  
# 03. 전이 학습을 활용한 이미지 분류기 개선

## 과제 설명 및 요구사항
  + 설명
     + 사전 학습된 모델(예: VGG16)을 활용하여 전이 학습을 수행하고, 이미지 분류기의 성능을 향

  + 요구사항
      + tensorflow.keras.applications에서 사전 학습된 VGG16 모델을 로드하고, 최상위 레이어 제거
      + 새로운 데이터셋에 맞게 추가 레이어를 설계
      + 사전 학습된 모델의 가중치를 고정한 후, 새로운 레이어를 추가하여 훈련시켜 전이 학습 수행
      + 모델 훈련 및 성능 평가 
      + 기존 모델과의 성능 비교
         
## 전체 코드 
   ```
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
 ```

## 데이터 로드 및 전처리 
 ```
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  
 ```
+ cifar10.load_data()를 사용해 데이터셋 로드 
+ 0~255 픽셀 값을 0~1 범위로 정규화(Min-Max Sacling)
  
## 전이 학습 모델 구성 
 ```
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
 ```
+ VGG16 : ImageNet 가중치 사용 (1000개 클래스 사전 학습)
+ 가중치를 고정하여 특징 추출기 역할만 수행하도록 설정
+ Flatten, Dense, Dropout 레이어를 사용해 새로운 데이터셋에 맞도록 설계 

## 기본 CNN 모델 구조 
 ```
baseline_model = models.Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
 ```
+ 2번에서 사용한 CNN 모델 사용 

## 모델 컴파일 및 훈련
 ```
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, 
                    epochs=10,
                    validation_split=0.2)

baseline_model.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

baseline_history = baseline_model.fit(x_train, y_train, 
                                      epochs=10,
                                      validation_split=0.2)
 ```

## 성능 비교 시각화 
 ```
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
 ```
+ 각 모델 Accuracy와 Loss에 대한 시각화
  
## 실행 결과 
<img src="https://github.com/user-attachments/assets/3976f086-f7e2-464f-966a-d2539a961196"/>

+ VGG16 테스트 정확도: 0.6053
+ CNN 테스트 정확도: 0.6834
