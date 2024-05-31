import numpy as np
import cv2
import streamlit as st
import re
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import requests
from localStoragePy import localStoragePy
import tensorflow as tf
import logging

# Enable logging for debugging
logging.basicConfig(level=logging.DEBUG)

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

localStorage = localStoragePy('my_app', 'json')

model = "@cf/meta/llama-2-7b-chat-int8"
account_id = "dde57a514ddb8385b5c01fdadc5f78b7"
api_token = "JiJkejsqQ888ICVLwuUpOJF-pbnC5dXurOC1iXHD"  

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

st.title('Emotion Powered Chatbot')

def format_response(response):
    emoji_map = {
        "Oh, wow!": "\n😮",
        "giggles": "\n😄",
        "Oh, my goodness!": "\n😲",
        "blinks": "\n😉",
        "nervous chuckle": "\n😅",
        "excitedly": "\n😃",
        "Yes, that's right!": "\n👍",
        "nods": "\n👌",
        "smiling": "\n😊",
        "bounces up and down": "\n😁",
    }

    for phrase, emoji in emoji_map.items():
        response = response.replace(phrase, emoji)

    response = re.sub(r'\s*\([^)]*\)', '', response)

    return response

class VideoTransformer(VideoTransformerBase):
    frame_count = 0
    last_faces = None
    last_predictions = None

    gender_net = cv2.dnn.readNetFromCaffe('gender_deploy.prototxt', 'gender_net.caffemodel')
    age_net = cv2.dnn.readNetFromCaffe('age_deploy.prototxt', 'age_net.caffemodel')
    facecasc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    gender_list = ['Male', 'Female']

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    model.load_weights('pr_model.h5')

    def transform(self, frame):
        MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)

        frame = frame.to_ndarray(format="bgr24")

        if VideoTransformer.frame_count % 5 == 0:
            facecasc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            VideoTransformer.last_faces = faces

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                prediction = VideoTransformer.model.predict(cropped_img)
                maxindex = int(np.argmax(prediction))
                VideoTransformer.last_predictions = (maxindex, prediction)

        else:
            faces = VideoTransformer.last_faces
            maxindex, prediction = VideoTransformer.last_predictions

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            gender_list = ['Male', 'Female']

            roi = cv2.resize(frame[y:y+h, x:x+w], (227, 227))
            blob = cv2.dnn.blobFromImage(roi, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            VideoTransformer.gender_net.setInput(blob)
            gender_preds = VideoTransformer.gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]

            age_list = ['0-2', '3-8', '9-15', '16-25', '26-35', '36-45', '46-60', '61+']

            roi = cv2.resize(frame[y:y+h, x:x+w], (227, 227))
            blob = cv2.dnn.blobFromImage(roi, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            VideoTransformer.age_net.setInput(blob)
            age_preds = VideoTransformer.age_net.forward()
            age = age_list[age_preds[0].argmax()]

            cv2.putText(frame, f"{gender}, {age}, {emotion_dict[maxindex]}", (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            localStorage.setItem('key', maxindex)

        VideoTransformer.frame_count += 1

        return frame

activities = ["Home", "Webcam Face Detection", "About"]
choice = st.sidebar.selectbox("Select Activity", activities)

if choice == "Home":
    html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
        <h4 style="color:white;text-align:center;">
        Gender, Age and Emotion detection application using OpenCV, Custom CNN model and Streamlit.</h4>
        </div>
        </br>"""
    st.markdown(html_temp_home1, unsafe_allow_html=True)
    st.write("""
        The application has two functionalities.

        1. Real time face detection using web cam feed.

        2. Real time face emotion recognization.
        """)
elif choice == "Webcam Face Detection":
    st.header("Webcam Live Feed")
    st.write("Click on start to use webcam and detect your face emotion")

    rtc_config = RTCConfiguration({
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun3.l.google.com:19302"]},
            {"urls": ["stun:stun4.l.google.com:19302"]},
        ]
    })

    webrtc_ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        video_processor_factory=VideoTransformer,
        media_stream_constraints={"video": True, "audio": False}
    )

    user_input = st.text_input("Ask something:")
    if st.button('Send'):
        if user_input != '':
            value = localStorage.getItem('key')

            response = requests.post(
                f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}",
                headers={"Authorization": f"Bearer {api_token}"},
                json={"messages": [
                    {"role": "system", "content": f"You are an emotion powered chatbot. Your responses are influenced by the user emotions. Currently the user is {emotion_dict[int(value)]}!"},
                    {"role": "user", "content": user_input}
                ]}
            )

            inference = response.json()
            if 'result' in inference:
                formatted_response = format_response(inference["result"]["response"])
                st.markdown(formatted_response)
            else:
                st.error(f"API request failed. Error: {inference['errors']}")
        else:
            st.write('Please enter a question.')

elif choice == "About":
    st.subheader("About this app")
    html_temp_about1 = """<div style="background-color:#6D7B8D;padding:10px">
        <h4 style="color:white;text-align:center;">
        Real time face emotion detection application using OpenCV, Custom Trained CNN model and Streamlit.</h4>
        </div>
        </br>"""
    st.markdown(html_temp_about1, unsafe_allow_html=True)
