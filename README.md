# Weather Classification ⛅🌧☁

## Technologies 💡
![Tensor Flow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-FF0000?style=for-the-badge&logo=keras&logoColor=white)
![PyCharm](https://img.shields.io/badge/pycharm-143?style=for-the-badge&logo=pycharm&logoColor=black&color=black&labelColor=green)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

## Description ❔❓

In this project I use images depicting weather phenomena to image classification. I use TensorFlow and EfficientNet B0 pretrained model. 

For the time being I predict a weather phenomenon for four pictures:

#### - Rain
<img src="https://user-images.githubusercontent.com/122997699/219761127-3bf7a322-b407-439f-9ce6-12e66270b87a.jpg" width="500" height="500">

#### - Cloudy
<img src="https://user-images.githubusercontent.com/122997699/219760695-a9dcd1f4-e28d-44e5-9fe5-c02e18c91f7b.jpg" width="500" height="500">

#### - Sunrise
<img src="https://user-images.githubusercontent.com/122997699/219761626-92e1cdca-670f-4967-926c-2ee9171d421d.jpg" width="500" height="500">

#### - Shine
<img src="https://user-images.githubusercontent.com/122997699/219761633-9421f6fc-b297-4ce2-9bb0-b841ac55a06a.jpg" width="500" height="500">


## Dataset📁
Images used in this project you can find [here](https://data.mendeley.com/datasets/4drtyfjtfy/1)

This datasets contain images depicting four weather phenomena such as:
* Rain
* Clody
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
![image](https://user-images.githubusercontent.com/122997699/219766017-abc15ad2-ed5a-48d5-bece-a1782ac894d9.png)



## Next goals 🏆⌛
#### * Increasing the test dataset
#### * Accuracy check for a larger set of test data
#### * Added new weather phenomena
