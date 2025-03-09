import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np

# Load Model
model = load_model(r"D:\image_classfication\Image_classify.keras")

# Categories
data_cat = ['Butter fruit', 'Cucumber Country', 'Dragon Fruit', 'Muskmelon', 'Strawberry', 
            'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 
            'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 
            'grapes', 'green apple', 'jalepeno', 'kiwi', 'koyya', 'lemon', 'lettuce', 'mango', 
            'onion', 'orange', 'papaya', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 
            'potato', 'raddish', 'red bannan', 'sapota', 'soy beans', 'spinach', 'sweet lime', 
            'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']

# Image size
img_width, img_height = 180, 180

# Load Image
image_path = r"D:\image_classfication\Fruits_Vegetables\ko1.png"
image_load = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
img_arr = tf.keras.utils.img_to_array(image_load)
img_bat = tf.expand_dims(img_arr, 0)  # Add batch dimension

# Predict
predict = model.predict(img_bat)
score = tf.nn.softmax(predict[0])  # Apply softmax

# Display results in Streamlit
st.image(image_load, caption="Uploaded Image", width=150)
st.write("Veg/Fruit in image is {} with accuracy of {:0.2f}%".format(
    data_cat[np.argmax(score)], np.max(score) * 100))
