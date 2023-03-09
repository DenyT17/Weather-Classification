# Weather Classification ⛅🌧☁

## Technologies 💡
![Tensor Flow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-FF0000?style=for-the-badge&logo=keras&logoColor=white)
![PyCharm](https://img.shields.io/badge/pycharm-143?style=for-the-badge&logo=pycharm&logoColor=black&color=black&labelColor=green)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

## Description ❔❓

In this project I use images depicting weather phenomena to classifier image phenomena on it. I use TensorFlow and EfficientNet(B0 - B4) to  pretrained models. 

When I will add few new classes and evaluate accuracy will be satisfactory, I will want to predict weather phenomena in images from YouTube videos which showing the landscape.  

For the time being I predict this weather phenomenon :
- Rain
- Cloudy
- Sunrise
- Shine
- Snow


## Dataset📁
Images used in this project you can find [here](https://data.mendeley.com/datasets/4drtyfjtfy/1)

This datasets contain images depicting four weather phenomena such as:
* Rain
* Cloudy
* Sunrise
* Sunshine

Pictures of snow I found and downloaded myself from the Internet.

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

#### 5️⃣ Now I must compile,train and save my model: 
``` python
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=["accuracy"]
    )
epochs = 10
hist = model.fit(ds_train, epochs=epochs, validation_data=ds_val, verbose=2)  
``` 


I used EfficentNet pretrained models from B0 to B4, below are the prediction results for the test data:
![image](https://user-images.githubusercontent.com/122997699/220909234-5bd81bb2-6411-4fef-890b-89a807be21ca.png)

Each of pretrained models, give accucary above 90 %. 

## Testing images extracted from YouTube videos. 

In this case I must use PyTube and OpenCv2 libraries, to have the opportunities for download video from Youtube.
I create two function: 
* Download - thanks to this function I can download video from Youtube and save it in chosen directory.
```python
def Download(link,path):
    youtubeObject = YouTube(link)
    youtubeObject = youtubeObject.streams.get_highest_resolution()
    youtubeObject.download(path)
    print("Download is completed successfully, video : {0} has been saved in {1}"
          .format(youtubeObject.title,path))
```
* Get_Frame - this function extract images from video, and save them in specified directory. I added functionality thanks to witch I can specified how many images per second will be extracted.
```python
def Get_Frame(path,img_name,img_per_s):
    cam = cv2.VideoCapture(path)
    # Calculation of video duration
    fps = cam.get(cv2.CAP_PROP_FPS)
    frame_count = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    # Creating new folder from images, if it doesn't exist
    if not os.path.exists('test_img_YT/'+img_name):
        os.makedirs('test_img_YT/'+img_name)
        
    currentframe = 1
    img_number=0

    # Saving chosen number of images per second
    while (True):
        ret, frame = cam.read()

        if ret:
            if currentframe % (30/img_per_s) ==0:
                name = 'test_img_YT/'+img_name+'/'+ img_name + str(img_number) + '.jpg'
                print('Creating...' + name)
                cv2.imwrite(name, frame)
                img_number+=1
            currentframe += 1
        else:
            break
    cam.release()
    cv2.destroyAllWindows()
```
Both of this function are in YoutubeDownload.py file. 

#### In Class_img_for_YT files I download videos and extract images from them. Next I make predict on weather phenomena on this images. Unfortunately in this moments accuracy of prediction very small so I will must modify my model.  

## Next goals 🏆⌛
#### * Added new weather phenomena.
#### * Increase accuracy from test image from Youtube videos.
