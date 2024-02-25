import streamlit as st
from keras.applications.vgg16 import VGG16, decode_predictions
from keras.preprocessing import image
from PIL import Image
import numpy as np
import pandas as pd
import boto3
import os

# prepare learned CNN model
model = VGG16(weights="imagenet", include_top=True)
model.summary()

# show title
st.title('EdgeAI vs CloudAI')

# image input
edge_uploaded_file = st.file_uploader("Recognition by EdgeAI", type='jpg', key='edge_key')

if edge_uploaded_file:
    # get input
    input_img = Image.open(edge_uploaded_file).resize((224, 224))

    # prepare img
    preprocessed_img = np.stack([image.img_to_array(input_img)])

    # predict
    results = decode_predictions(model.predict(preprocessed_img), top=10)

    # show bar chart
    st.write("recognition result (%)")
    data = pd.DataFrame(
        [row[2] * 100 for row in results[0]],
        index=[row[1] for row in results[0]]
    )
    st.bar_chart(data)

# image input
cloud_uploaded_file = st.file_uploader("Recognition by CloudAI", type='jpg', key='cloud_key')

# 　Specifying aws services
client = boto3.client(
    'rekognition',
    aws_access_key_id=os.environ.get("aws_access_key_id"),
    aws_secret_access_key=os.environ.get("aws_secret_access_key"),
    region_name='ap-northeast-1'
)

if cloud_uploaded_file:
    # get input
    input_img = Image.open(cloud_uploaded_file).resize((224, 224))
    input_img.save("input_image.png")
    with open("input_image.png", 'rb') as image:
        response = client.detect_labels(Image={'Bytes': image.read()})
    st.write('show result')
    for label in response['Labels']:
        st.write(label['Name'] + ' : ' + str(label['Confidence']))
