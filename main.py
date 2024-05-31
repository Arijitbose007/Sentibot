import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

class SimpleVideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Title of the app
st.title("WebRTC Test with Debugging")

# WebRTC streamer setup
webrtc_ctx = webrtc_streamer(key="example", video_transformer_factory=SimpleVideoTransformer)

# Debugging output
if webrtc_ctx.state.playing:
    st.write("WebRTC is playing")
else:
    st.write("WebRTC is not playing")

# Check if the webrtc context is correctly initialized
if webrtc_ctx.video_transformer:
    st.write("Video transformer is initialized")
else:
    st.write("Video transformer is not initialized")

# Display any errors encountered
if webrtc_ctx.state.error:
    st.write(f"Error: {webrtc_ctx.state.error}")

# Additional debug information
st.write(f"WebRTC state: {webrtc_ctx.state}")
st.write(f"WebRTC config: {webrtc_ctx.config}")
