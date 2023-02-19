# Weather Classification ⛅🌧☁

## Technologies 💡
![Tensor Flow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-FF0000?style=for-the-badge&logo=keras&logoColor=white)
![PyCharm](https://img.shields.io/badge/pycharm-143?style=for-the-badge&logo=pycharm&logoColor=black&color=black&labelColor=green)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

## Description ❔❓

In this project I use images depicting weather phenomena to image classification. I use TensorFlow and EfficientNet B0 pretrained model. 

For the time being I predict a weather phenomenon for twelf pictures, three for each weather phenomena. 


#### - Rain
<img src="https://user-images.githubusercontent.com/122997699/219761127-3bf7a322-b407-439f-9ce6-12e66270b87a.jpg" width="250" height="250"/>  <img src="https://user-images.githubusercontent.com/122997699/219870248-fbb5e6dd-0ad7-42ad-9faf-35ce52ff3921.jpg" width="250" height="250"/>  <img src="https://user-images.githubusercontent.com/122997699/219870251-7174a839-4d0f-4390-961e-643c7a98d59b.png" width="250" height="250"/>

#### - Cloudy
<img src="https://user-images.githubusercontent.com/122997699/219870437-b6da9be4-2970-4145-8688-23e7b13080d7.jpg" width="250" height="250"/>  <img src="https://user-images.githubusercontent.com/122997699/219870440-947f8572-9787-4d36-80eb-713f48c57d24.jpg" width="250" height="250"/>  <img src="https://user-images.githubusercontent.com/122997699/219870441-72ddabe8-7efd-47cd-93ee-4fcf2af2c6cd.jpg" width="250" height="250"/>

#### - Sunrise
<img src="https://user-images.githubusercontent.com/122997699/219870641-607e9624-b390-4d26-a678-faf7a3fb870f.jpg" width="250" height="250"/>  <img src="https://user-images.githubusercontent.com/122997699/219870640-5373ca06-2438-4d01-92c7-4ae8726f2a4f.jpg" width="250" height="250"/>  <img src="https://user-images.githubusercontent.com/122997699/219870639-783929c6-1471-4eae-b9bd-f77e2ab37700.jpg" width="250" height="250"/>

#### - Shine
<img src="https://user-images.githubusercontent.com/122997699/219870608-e6547135-3b33-4e12-8f20-f068e5212384.jpg" width="250" height="250"/>  <img src="https://user-images.githubusercontent.com/122997699/219870607-5aebbeb2-10f8-4a8b-af8e-c3ce3118cf95.jpg" width="250" height="250"/>  <img src="https://user-images.githubusercontent.com/122997699/219870606-380dce43-7a5c-46b9-ad42-58b1f9651424.jpg" width="250" height="250"/>


## Dataset📁
Images used in this project you can find [here](https://data.mendeley.com/datasets/4drtyfjtfy/1)

This datasets contain images depicting four weather phenomena such as:
* Rain
* Cloudy
* Sunrise
* Sunshine

I separated the photos into appropriate directories which I named accordingly. In futur i want extend the dataset with other weather phenomena.

## The method of project implementation ✅

A detailed description of the implementation of the project can be found in the Weather Classification.py file

#### 1️⃣ First I load and spli data: 
```python
train = tf.keras.utils.image_dataset_from_directory(r'Weather Classification\Weather',
                                                                 shuffle=True,
                                                                 batch_size=batch_size,
                                                                 image_size=(img_size,img_size))
                                                                 
train_batches = tf.data.experimental.cardinality(train)
val = train.take(train_batches // 5)
train = train.skip(train_batches // 5)
```
#### 2️⃣ Because my dataset is pretty small, I must augmentation it:
```python
img_augmentation = Sequential(
    [
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip(),
        layers.RandomContrast(factor=0.1)
    ],
    name="img_augmentation",
)
```

#### 3️⃣ Now I define pretrained model and global average pooling layer: 
``` python
image_batch, label_batch = next(iter(ds_train))
feature_batch = base_model(image_batch)
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
```

#### 4️⃣ In next step I can combine all layers to model: 
``` python
inputs = tf.keras.Input(shape=(img_size,img_size,3))
x = img_augmentation(inputs)
x = base_model(x)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)
model = tf.keras.Model(inputs, outputs)
model.summary()
``` 

Model summary:

![image](https://user-images.githubusercontent.com/122997699/219765405-9f8cd4f7-d327-46fd-8ff3-287c452378d6.png)

#### 5️⃣ Now I must compile and train my model: 
``` python
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=["accuracy"]
    )
epochs = 10
hist = model.fit(ds_train, epochs=epochs, validation_data=ds_val, verbose=2)  
``` 
![image](https://user-images.githubusercontent.com/122997699/219765706-629efddb-8cfc-4172-949e-1fed3392ec0b.png)

Accuracy plot depending on epochs: 

![Figure_1](https://user-images.githubusercontent.com/122997699/219765766-351b590c-b667-46fe-ae68-44ec1994406a.png)

#### 6️⃣ At the end I can make prediction of weather phenomena in test images: 
``` python
prediction=model.predict(test)
for i in range(test_img_count):
    score = prediction[i]
    print("The weather in the picture looks like: {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score)))
    plt.figure(figsize=(5, 5))
    for images, labels in test.take(1):
        ax = plt.subplot()
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[np.argmax(score)])
``` 

Classification results

![image](https://user-images.githubusercontent.com/122997699/219870799-8a816451-c05c-40d3-ac89-26d5a3067816.png)

Test photos with classification in the title of the chart.

![Figure_1](https://user-images.githubusercontent.com/122997699/219870950-16c6c714-5cd2-44f8-8b23-4c466d18dcc1.png)
![Figure_2](https://user-images.githubusercontent.com/122997699/219870951-a85c1c95-886f-4d83-962d-6808ecaf4d07.png)
![Figure_3](https://user-images.githubusercontent.com/122997699/219870952-ccebf1d6-0d3f-4169-9d8a-f0160e609d50.png)
![Figure_4](https://user-images.githubusercontent.com/122997699/219870953-e63a4c14-da78-42a4-8cdd-9c94f61a3d31.png)
![Figure_5](https://user-images.githubusercontent.com/122997699/219870955-bd515fac-9437-4879-b625-ad580d437d33.png)
![Figure_6](https://user-images.githubusercontent.com/122997699/219870956-7373dec3-c99e-45a0-9bbb-50df1024c448.png)
![Figure_7](https://user-images.githubusercontent.com/122997699/219871243-f3f3803a-bc5a-47e8-9b27-67410b772913.png)
![Figure_8](https://user-images.githubusercontent.com/122997699/219870957-306f9ca8-7e4c-48a4-9500-fa573342c5e0.png)
![Figure_9](https://user-images.githubusercontent.com/122997699/219870958-54d2b0d4-6ee5-49d4-b5ac-1f1a9893bf55.png)
![Figure_10](https://user-images.githubusercontent.com/122997699/219870960-65194ec2-71d4-4c1b-9a6a-63dd86cbb7fa.png)
![Figure_11](https://user-images.githubusercontent.com/122997699/219870961-f6ce2cc1-a762-42e4-8fb6-4c9b226354e1.png)
![Figure_12](https://user-images.githubusercontent.com/122997699/219870962-8ef92e18-cf85-4b5b-b74a-f027a6acca79.png)




In this case image classification is correct. 

## Next goals 🏆⌛
#### * Increasing the test dataset
#### * Accuracy check for a larger set of test data
#### * Added new weather phenomena
