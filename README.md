Sentibot 🤖
This emotion-powered chatbot uses Streamlit for deployment and OpenCV for an emotion detector (as well as gender and age detection)to generate emotion influenced responses to help the user accordingly.

Run Locally
Clone this Git Repository:
git clone https://github.com/aroproduction/emotion_powered_chatbot
Change the working directory to the cloned directory:
cd emotion_powered_chatbot
The customly-trained model trained on "fer2013.csv" dataset (https://www.kaggle.com/datasets/deadskull7/fer2013) with around 63.41% emotion detection accuracy is given as "pr_model.h5".

Run
pip install -r requirements.txt
to install the dependencies.

Finally to run the main page:
streamlit run main.py

Screenshots
Home Page
App Screenshot

Detection Page
App Screenshot

Live Chat based on detection
App Screenshot

