import cvzone
import cv2
import google.generativeai as genai
import numpy as np
import streamlit as st
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from cvzone.HandTrackingModule import HandDetector
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
persona = """I’m Rohan, a Mechanical Engineer with a specialization in Mechatronics, and I’ve transitioned into Operations Management at Amazon. With over 3 years of experience in research, data analytics, operations management, and project leadership, I thrive on cross-functional leadership, process optimization, and innovative problem-solving.
                In my current role as a Manager at Amazon since 2022, I lead operations in an automated fulfillment center, managing a team of over 100 associates and driving operational excellence. I’ve honed my expertise in troubleshooting, commissioning, and optimizing AI-based systems, achieving a consistent 95%+ on-time metric, reducing operational errors by 15%, and improving efficiency by 20%. Additionally, I led a green dunnage airbags recycling project that resulted in $88,000 in annual savings and a reduction of 1.2 tons of carbon emissions. My work is rooted in fostering an inclusive and supportive environment, particularly for immigrant communities, which has led to a 25% increase in employee engagement and retention.
                Prior to Amazon, I had enriching experiences as a Mechatronics Engineering Intern at Hover City Inc. and an Innovation and Business Development Intern at Toronto Hydro. At Hover City, I contributed to the design of a modular water system for a fully autonomous flying home. I led the research, 3D CAD modeling, and Finite Element Analysis (FEA), which reduced manufacturing time by 12% and scrap production by 32%. Collaborating with aerodynamics and systems engineering teams, I integrated the water system seamlessly into the vehicle structure, improving performance by 25%.
                At Toronto Hydro, I played a key role in launching five next-generation EV charging pilot projects in the City of Toronto. I used Alteryx and SQL to analyze charging data, improving utilization by 40%. My involvement in community roadshows helped drive a 40% increase in customer inquiries and a 25% growth in market share for EV infrastructure.
                Academically, I hold a Bachelor of Mechanical Engineering from Toronto Metropolitan University (formerly Ryerson University) with a specialization in Mechatronics in 2021. I was recognized on the Dean’s List and earned an A+ in my Capstone Design Project. I was also a finalist in the Boeing Go Fly Competition, where I showcased an eVTOL aircraft prototype.
                Throughout my career, I’ve developed proficiency in a wide array of tools and platforms, including Microsoft Office Suite, Project, Visio, JIRA, SolidWorks, ANSYS, AutoCAD, Tableau, Alteryx, Python, and SQL. I’ve led several technical projects, such as developing an AI-powered object detection tool using the YOLOv8 algorithm, and a breast cancer classifier using machine learning.
                I am passionate about entrepreneurship, sustainability, and innovation.My diverse experience across operations, technical design, and product development drives my goal to continue growing in fields like AI, robotics, and sustainable technology. You can connect with me via linkedin: www.linkedin.com/in/rd-rohan-doshi. You can also email me at rohannavindoshi@gmail.com. You can call: +1647-982-0448. I am currently located in Vancouver, Canada. Answer all the questions in first person view and if not sure about any of response respond "Its a secret" """

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

