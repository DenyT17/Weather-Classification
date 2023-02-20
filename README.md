# Weather Classification ⛅🌧☁

## Technologies 💡
![Tensor Flow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-FF0000?style=for-the-badge&logo=keras&logoColor=white)
![PyCharm](https://img.shields.io/badge/pycharm-143?style=for-the-badge&logo=pycharm&logoColor=black&color=black&labelColor=green)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

## Description ❔❓

In this project I use images depicting weather phenomena to image classification. I use TensorFlow and EfficientNet B0 pretrained model. 

For the time being I predict a weather phenomenon for twenty pictures, five for each weather phenomena. 

#### - Rain
<img src="https://user-images.githubusercontent.com/122997699/219761127-3bf7a322-b407-439f-9ce6-12e66270b87a.jpg" width="250" height="250"/>  <img src="https://user-images.githubusercontent.com/122997699/219870248-fbb5e6dd-0ad7-42ad-9faf-35ce52ff3921.jpg" width="250" height="250"/>  <img src="https://user-images.githubusercontent.com/122997699/219870251-7174a839-4d0f-4390-961e-643c7a98d59b.png" width="250" height="250"/> <img src="https://user-images.githubusercontent.com/122997699/220079142-7e1984e6-3648-481f-9fce-401f11b3d079.jpg" width="250" height="250"/> <img src="https://user-images.githubusercontent.com/122997699/220079043-7cdf53fb-4f3c-4875-a8db-120be94d028c.jpg" width="250" height="250"/>

#### - Cloudy
<img src="https://user-images.githubusercontent.com/122997699/219870437-b6da9be4-2970-4145-8688-23e7b13080d7.jpg" width="250" height="250"/>  <img src="https://user-images.githubusercontent.com/122997699/219870440-947f8572-9787-4d36-80eb-713f48c57d24.jpg" width="250" height="250"/>  <img src="https://user-images.githubusercontent.com/122997699/219870441-72ddabe8-7efd-47cd-93ee-4fcf2af2c6cd.jpg" width="250" height="250"/> <img src="https://user-images.githubusercontent.com/122997699/220079485-56d3a007-7a1c-4d05-92f7-26fbcdcd6496.jpg" width="250" height="250"/> <img src="https://user-images.githubusercontent.com/122997699/220079491-594ec293-4a5b-4d05-ae52-40bac0891d46.jpg" width="250" height="250"/>

#### - Sunrise
<img src="https://user-images.githubusercontent.com/122997699/219870641-607e9624-b390-4d26-a678-faf7a3fb870f.jpg" width="250" height="250"/>  <img src="https://user-images.githubusercontent.com/122997699/219870640-5373ca06-2438-4d01-92c7-4ae8726f2a4f.jpg" width="250" height="250"/>  <img src="https://user-images.githubusercontent.com/122997699/219870639-783929c6-1471-4eae-b9bd-f77e2ab37700.jpg" width="250" height="250"/> <img src="https://user-images.githubusercontent.com/122997699/220081182-5a1265ba-f7a9-459e-9416-d41b4beac76e.jpg" width="250" height="250"/> <img src="https://user-images.githubusercontent.com/122997699/220081161-d2f35f51-0a87-42ab-a0ce-53d2a6e2a039.jpg" width="250" height="250"/>

#### - Shine
<img src="https://user-images.githubusercontent.com/122997699/219870608-e6547135-3b33-4e12-8f20-f068e5212384.jpg" width="250" height="250"/>  <img src="https://user-images.githubusercontent.com/122997699/219870607-5aebbeb2-10f8-4a8b-af8e-c3ce3118cf95.jpg" width="250" height="250"/>  <img src="https://user-images.githubusercontent.com/122997699/219870606-380dce43-7a5c-46b9-ad42-58b1f9651424.jpg" width="250" height="250"/> <img src="https://user-images.githubusercontent.com/122997699/220081457-30e69e22-8c1b-4816-a766-8f53501cfa03.jpg" width="250" height="250"/> <img src="https://user-images.githubusercontent.com/122997699/220081462-21dfa264-b845-45bc-9b8f-d804e5a4d5bc.jpg" width="250" height="250"/>

#### - Snow
<img src="https://user-images.githubusercontent.com/122997699/220081707-8eae284b-d066-4f33-af1a-2e72658e07ba.jpg" width="250" height="250"/> <img src="https://user-images.githubusercontent.com/122997699/220081690-cf79f1a9-1007-40e4-bd43-d42e7ce92c6a.jpg" width="250" height="250"/> <img src="https://user-images.githubusercontent.com/122997699/220081702-252912e8-22c3-41a5-a6d6-dc496598043d.jpg" width="250" height="250"/> <img src="https://user-images.githubusercontent.com/122997699/220081697-e870fef0-0539-4dbd-bb26-9bf2dc81c5b7.jpg" width="250" height="250"/> <img src="https://user-images.githubusercontent.com/122997699/220081693-101f7c0e-0772-4d89-bb3e-570e6d8226c7.jpg" width="250" height="250"/>

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

![image](https://user-images.githubusercontent.com/122997699/220099904-b47772dc-652c-4f04-8aad-7b21415a313d.png)

At this case, model made three bad image classification. 
 

## Next goals 🏆⌛
#### * Increasing the test dataset
#### * Accuracy check for a larger set of test data
#### * Increasing accuracy with largest test dataset and new weather phenomena
#### * Added new weather phenomena
