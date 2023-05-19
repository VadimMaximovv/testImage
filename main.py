import numpy as np # linear algebra
import os
import cv2
import tensorflow as tf
import keras
import keras.utils as image
import keras.utils as img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten,Dense,Dropout,BatchNormalization
from keras import regularizers
from keras.optimizers import Adam


#model = tf.keras.models.load_model('model_optimal.h5')

#model.load_weights('model_weights.h5')
#img = image.load_img("im14.png", target_size=(48, 48), color_mode="grayscale")
#img = np.array(img)

#label_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy ', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
#img = np.expand_dims(img, axis=0) #makes image shape (1,48,48)
#img = img.reshape(1, 48, 48, 1)
#result = model.predict(img)
#result = list(result[0])
#img_index = result.index(max(result))
#print(label_dict[img_index])
#img = tf.keras.utils.load_img('C:/Users/vadim/OneDrive/Документы/Sharga/Семестр 4/ИИИИИИИИИИИИИ/проект/Emotion/test/sad/im37.png', target_size=(48, 48))
"""
model = tf.keras.models.Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (5, 5), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(7, activation='softmax'))

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
"""
def prepare(filepath):
    IMG_SIZE = 48  # 50 in txt-based
    img_array = cv2.imread(filepath)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    new_array = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    new_array = new_array.astype("float") / 255.0
    return new_array

model = tf.keras.models.load_model('emotion_detection.h5')

CATEGORIES = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
prediction = model.predict([prepare("im12.png")])
print(prediction)  # will be a list in a list.
score = tf.nn.softmax(prediction[0])
print(score)
print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(CATEGORIES[np.argmax(score)], 100 * np.max(score)))