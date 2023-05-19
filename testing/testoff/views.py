from django.shortcuts import render
from .forms import ImageForm
import numpy as np
import tensorflow as tf
import keras.utils as image
import cv2

def prepare(filepath):
    IMG_SIZE = 48
    img_array = cv2.imread(filepath)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    new_array = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    new_array = new_array.astype("float") / 255.0
    return new_array

def image_upload_view(request):
    """Process images uploaded by users"""
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            # Get the current instance object to display in the template
            img_obj = form.instance
            asd = str(request.FILES)[51:-16]

            model = tf.keras.models.load_model('./models/emotion_detection.h5')
            #img = image.load_img("./media/images/" + asd, target_size=(48, 48), color_mode="rgb")
            CATEGORIES = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
            prediction = model.predict([prepare("./media/images/" + asd)])
            prediction  # will be a list in a list.
            score = tf.nn.softmax(prediction[0])
            answer = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(CATEGORIES[np.argmax(score)], 100 * np.max(score))
            #return render(request, 'index.html', {'form': form, 'img_obj': img_obj})
            return render(request, 'index.html', {'form': form, 'img_obj': img_obj, 'answer': answer})
    else:
        form = ImageForm()
    return render(request, 'index.html', {'form': form})