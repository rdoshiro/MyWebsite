import cvzone
import cv2
import google.generativeai as genai
import numpy as np
import streamlit as st
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from PIL import Image

# Configure Google Generative AI
api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# Header and introduction
col1, col2 = st.columns(2)
with col1:
    st.title("Hi, I'm Rohan Doshi")

with col2:
    st.image("Images/image (1).jpg")

st.title(" ")

# Define your persona
persona = """I’m Rohan, a Mechanical Engineer..."""

# Chat interface with virtual persona
st.title("Chat with virtual me")
user_question = st.text_input("Ask anything you would like to know about me?")

if st.button("ASK"):
    prompt = persona + user_question
    response = model.generate_content(prompt)
    st.write(response.text)

# Projects gallery
st.title("Projects Gallery")
col5, col6 = st.columns([4, 4])
with col5:
    st.subheader("Interactive Gesture Control Map")
    st.video("Videos/Interactive_Map.mp4")
with col6:
    st.subheader("Autonomous Robotic Vehicle")
    st.video("Videos/Robotic_Vehicle.mov")

# Interactive AI content generator
st.subheader("Interactive AI Content Generator")
st.image("Images/HandSign.jpg", width=350)
st.subheader("Let's try it yourself")

col3, col4 = st.columns([2, 1])
with col3:
    run = st.checkbox('Run', value=False)

with col4:
    st.title("Answer")
    output_text_area = st.subheader("")

# Initialize Hand Detector from cvzone
detector = cvzone.HandTrackingModule.HandDetector(staticMode=False, maxHands=1, detectionCon=0.5)

# WebRTC video stream configuration
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# Transformer class to process the video stream
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.prev_pos = None
        self.canvas = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # Flip image horizontally for better hand tracking

        if self.canvas is None:
            self.canvas = np.zeros_like(img)

        hands, img = detector.findHands(img, flipType=False)  # Detect hand

        if hands:
            hand = hands[0]
            lmList = hand["lmList"]
            fingers = detector.fingersUp(hand)

            self.prev_pos, self.canvas = self.draw_hand_gesture(fingers, lmList, self.prev_pos, self.canvas)
            ai_text = self.send_to_ai(fingers)

            if ai_text:
                output_text_area.text(ai_text)

        # Merge video and canvas
        img_combined = cv2.addWeighted(img, 0.8, self.canvas, 0.2, 0)
        return img_combined

    def draw_hand_gesture(self, fingers, lmList, prev_pos, canvas):
        current_pos = None
        if fingers == [0, 1, 0, 0, 0]:  # Index finger up (draw)
            current_pos = lmList[8][0:2]  # Index finger tip
            if prev_pos is None:
                prev_pos = current_pos
            cv2.line(canvas, tuple(current_pos), tuple(prev_pos), (255, 0, 255), 10)
        elif fingers == [1, 1, 0, 0, 1]:  # Reset canvas gesture
            canvas = np.zeros_like(canvas)
        return current_pos, canvas

    def send_to_ai(self, fingers):
        if fingers == [1, 1, 1, 1, 1]:  # All fingers up (send to AI)
            pil_image = Image.fromarray(self.canvas)
            response = model.generate_content(["Guess the answer.", pil_image])
            return response.text
        return ""

# Start the WebRTC stream with hand tracking
webrtc_streamer(key="gesture-detection", video_transformer_factory=VideoTransformer, rtc_configuration=RTC_CONFIGURATION)

