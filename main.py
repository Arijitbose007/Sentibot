import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration
import av
import logging

logging.basicConfig(level=logging.DEBUG)

class SimpleVideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        return av.VideoFrame.from_ndarray(img, format="bgr24")

rtc_config = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]}
    ]
})

webrtc_ctx = webrtc_streamer(
    key="example",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_config,
    video_transformer_factory=SimpleVideoTransformer,
)

if webrtc_ctx.state.playing:
    st.write("WebRTC is playing")
else:
    st.write("WebRTC is not playing")

# Additional logging
st.write(f"WebRTC state: {webrtc_ctx.state}")
st.write(f"WebRTC context: {webrtc_ctx}")
