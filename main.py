import numpy as np
import cv2
import streamlit as st
import re
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, Flatten 
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
import requests
from localStoragePy import localStoragePy

import tensorflow as tf
import cv2
import numpy as np
import cv2
import numpy as np
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
    # Define mappings from phrases to emojis
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
    # Add more emojis here
    "phrase": "\nemoji",
    }

    # Replace phrases with emojis
    for phrase, emoji in emoji_map.items():
        response = response.replace(phrase, emoji)

     # Remove action descriptions
    response = re.sub(r'\s*\([^)]*\)', '', response)

    return response

 

class VideoTransformer(VideoTransformerBase):

    frame_count = 0  # Add a frame counter
    last_faces = None  # Store the last detected faces
    last_predictions = None  # Store the last predictions

   

     # Load the models and the Haar cascade classifier outside the transform function
    gender_net = cv2.dnn.readNetFromCaffe('gender_deploy.prototxt', 'gender_net.caffemodel')
    age_net = cv2.dnn.readNetFromCaffe('age_deploy.prototxt', 'age_net.caffemodel')
    facecasc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Define the gender list
    gender_list = ['Male', 'Female']

    # Create the model
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1), name="conv2d_1"))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', name="conv2d_2"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="maxpool2d_1"))
    model.add(Dropout(0.25, name="dropout_1"))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', name="conv2d_3"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="maxpool2d_2"))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', name="conv2d_4"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="maxpool2d_3"))
    model.add(Dropout(0.25, name="dropout_2"))

    model.add(Flatten(name="flatten"))
    model.add(Dense(1024, activation='relu', name="dense_1"))
    model.add(Dropout(0.5, name="dropout_3"))
    model.add(Dense(7, activation='softmax', name="dense_2"))

    # Load the model weights
    model.load_weights('pr_model.h5')

    def transform(self, frame):
        MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)

       


        # Find haar cascade to draw bounding box around face
        frame = frame.to_ndarray(format="bgr24")
         # Perform face detection every 5 frames
        if VideoTransformer.frame_count % 5 == 0:
            facecasc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
            VideoTransformer.last_faces = faces  # Store the detected faces

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                prediction = VideoTransformer.model.predict(cropped_img)
                maxindex = int(np.argmax(prediction))
                VideoTransformer.last_predictions = (maxindex, prediction)  # Store the predictions

        # Use the last detected faces and predictions for the other frames
        else:
            faces = VideoTransformer.last_faces
            maxindex, prediction = VideoTransformer.last_predictions

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            # Define the gender list
            gender_list = ['Male', 'Female']

            roi = cv2.resize(frame[y:y+h,x:x+w], (227, 227))
            blob = cv2.dnn.blobFromImage(roi, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            # Predict gender
            VideoTransformer.gender_net.setInput(blob)
            gender_preds = VideoTransformer.gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]

            # Preprocess the ROI for age detection
            age_list = ['0-2', '3-8', '9-15', '16-25', '26-35', '36-45', '46-60', '61+']  # Define the age list

            roi = cv2.resize(frame[y:y+h,x:x+w], (227, 227))
            blob = cv2.dnn.blobFromImage(roi, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            # Predict age
            VideoTransformer.age_net.setInput(blob)
            age_preds = VideoTransformer.age_net.forward()
            age = age_list[age_preds[0].argmax()]

            # Display the gender, age and emotion on the frame in a comma-separated format
            cv2.putText(frame, f"{gender}, {age}, {emotion_dict[maxindex]}", (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            localStorage.setItem('key', maxindex)

        VideoTransformer.frame_count += 1  # Increment the frame counter

        return frame

activiteis = ["Home", "Webcam Face Detection", "About"]
choice = st.sidebar.selectbox("Select Activity", activiteis)
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
    # Reduce the resolution of the video
    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer, client_settings={"video": {"width": 80, "height": 60}})

    user_input = st.text_input("Ask something:")
    if st.button('Send'):
        if user_input != '':
            value = localStorage.getItem('key')

            response = requests.post(
                f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}",
                headers={"Authorization": f"Bearer {api_token}"},
                json={"messages": [
                    {"role": "system", "content": f"You are a emotion powered chatbot. Your responses are influenced by the user emotions. Currently the user is  {emotion_dict[int(value)]}!"},
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
            st.chat_input('Please enter a question.')



elif choice == "About":
    st.subheader("About this app")
    html_temp_about1= """<div style="background-color:#6D7B8D;padding:10px">
                                <h4 style="color:white;text-align:center;">
                                Real time face emotion detection application using OpenCV, Custom Trained CNN model and Streamlit.</h4>
                                </div>
                                </br>"""
    st.markdown(html_temp_about1, unsafe_allow_html=True)

else:
    pass

# Remove a value from local storage
# localStorage.removeItem('key')

# Clear all values from local storage
