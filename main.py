import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

class SimpleVideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("WebRTC Test with Debugging")

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
st.write(f"WebRTC state: {webrtc_ctx.state}")
