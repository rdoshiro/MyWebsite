import cvzone
import cv2
import google.generativeai as genai
import numpy as np
import streamlit as st
from PIL import Image

# Configure the Google API
api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# App Layout
col1, col2 = st.columns(2)
with col1:
    st.title("Hi, I'm Rohan Doshi")
with col2:
    st.image("Images/image (1).jpg")

st.title(" ")

# Persona Information
persona = """   I’m Rohan, a Mechanical Engineer with a specialization in Mechatronics..."""
st.title("Chat with virtual me")

# Capture User Question
user_question = st.text_input("Ask anything you would like to know about me?")
if st.button("ASK", use_container_width=400):
    prompt = persona + user_question
    response = model.generate_content(prompt)
    st.write(response.text)

st.title("Projects Gallery")

# Project Videos
col5, col6 = st.columns([4, 4])
with col5:
    st.subheader("Interactive Gesture Control Map")
    st.video("Videos/Interactive_Map.mp4")
with col6:
    st.subheader("Autonomous Robotic Vehicle")
    st.video("Videos/Robotic_Vehicle.mov")

st.subheader("Interactive AI Content Generator")
st.image("Images/HandSign.jpg", width=350)
st.subheader("Let's try it yourself")

# Hand Detection Section
col3, col4 = st.columns([2, 1])
with col3:
    run = st.checkbox('Run', value=True)
    FRAME_WINDOW = st.image([])

with col4:
    st.title("Answer")
    output_text_area = st.subheader("")

# Initialize HandDetector
from cvzone.HandTrackingModule import HandDetector
cap = cv2.VideoCapture(0)  # Capture from webcam
cap.set(2, 480)
cap.set(2, 480)

detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)

# Variables for drawing
prev_pos = None
canvas = None
output_text = ""

def getHandInfo(img):
    hands, img = detector.findHands(img, draw=True, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    else:
        return None

def draw(info, prev_pos, canvas):
    fingers, lmlist = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]:
        current_pos = lmlist[8][0:2]
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, tuple(prev_pos), tuple(current_pos), (255, 0, 255), 10)
    elif fingers == [1, 1, 0, 0, 1]:  # Reset canvas
        canvas = np.zeros_like(canvas)
    return current_pos, canvas

def sendToAI(model, canvas, fingers):
    if fingers == [1, 1, 1, 1, 1]:
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["guess the answer.", pil_image])
        return response.text

# Streamlit Event Loop
if run:
    # Continuously get frames from the webcam
    success, img = cap.read()
    img = cv2.flip(img, flipCode=1)
    
    if canvas is None:
        canvas = np.zeros_like(img)
    
    info = getHandInfo(img)
    if info:
        fingers, lmlist = info
        prev_pos, canvas = draw(info, prev_pos, canvas)
        output_text = sendToAI(model, canvas, fingers)
    
    image_combined = cv2.addWeighted(img, 0.80, canvas, 0.20, 0)
    FRAME_WINDOW.image(image_combined, channels="BGR")
    
    if output_text:
        output_text_area.text(output_text)
