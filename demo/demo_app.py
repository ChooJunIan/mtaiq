import tensorflow as tf
import numpy as np
import streamlit as st
import cv2

from keras.applications.inception_resnet_v2 import preprocess_input
from keras import backend as K

def pearson_correlation(y_true, y_pred):
    # Subtract the mean from true and predicted values
    y_true_mean = K.mean(y_true)
    y_pred_mean = K.mean(y_pred)
    y_true_centered = y_true - y_true_mean
    y_pred_centered = y_pred - y_pred_mean

    # Calculate covariance and standard deviation
    covariance = K.mean(y_true_centered * y_pred_centered)
    y_true_std = K.std(y_true)
    y_pred_std = K.std(y_pred)

    # Calculate Pearson correlation coefficient
    pearson_coefficient = covariance / (y_true_std * y_pred_std + K.epsilon())

    return pearson_coefficient

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('/home/ian/Desktop/demo/model/multimodel_irnv2_weights_M1_demo/multimodel_irnv2_M1_demo.h5',
                                   custom_objects={"pearson_correlation": pearson_correlation},
                                   compile=False)
    return model

def predict_scores(image):
    resized_image = cv2.resize(image, (224, 224))
    preprocessed_image = preprocess_input(resized_image)
    input_data = np.expand_dims(preprocessed_image, axis=0)
    scores = model.predict([input_data] * 4)
    return scores

model = load_model()

def main():
    st.title("Multi-IRNV2 Demonstration Application")
    st.subheader("This is a simple demo for running the M1 model's Image Aesthetic Assessment and Image Quality Assessment.")
    st.markdown("Simply select an image, wait a few seconds, and the predicted aesthetic and quality scores of the image will be shown, ranging from 1-5, with 5 representing a high aesthetic/quality score.")


    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        st.image(image, channels="BGR")
        scores = predict_scores(image)

        # Display scores for all inputs 
        st.subheader("Aesthetic scores (1-5):")
        st.write(f"Aesthetic score 1: {round(scores[0][0].item(), 2)}")
        st.write(f"Aesthetic score 2: {round(scores[1][0].item(), 2)}")

        st.subheader("Quality scores (1-5):")
        st.write(f"Quality score 1: {round(scores[2][0].item(), 2)}")
        st.write(f"Quality score 2: {round(scores[3][0].item(), 2)}")  

if __name__ == '__main__':
    main() 