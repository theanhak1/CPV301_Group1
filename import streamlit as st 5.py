import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np

# Tiêu đề
st.title(" Camera ")

# Ô để hiển thị kết quả
result = st.empty()

# Màn hình để hiển thị video từ camera
st.header("Màn hình Camera")

if st.button("Tải dữ liệu từ ô kết quả"):
    st.write("Dữ liệu đã được tải: Hello, World!")

# Hiển thị video từ camera
webrtc_ctx = webrtc_streamer(key="example")
if webrtc_ctx.video_transformer:
    webrtc_ctx.video_transformer.video_source = cv2.VideoCapture(0)

# Hiển thị ô nhập kết quả
user_input = st.text_input("kết quả:")
if st.button("Xem kết quả"):
    result.text("Kết quả của bạn: " + user_input)
