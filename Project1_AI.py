import cvzone
#import cv2
import google.generativeai as genai
import numpy as np
import streamlit as st
import os
from PIL import Image

api_key = st.secrets["GOOGLE_API_KEY"]

genai.configure(api_key = api_key )
model = genai.GenerativeModel("gemini-1.5-flash")



col1, col2 = st.columns(2)
with col1:
    st.title("Hi, I'm Rohan Doshi")

with col2:
    st.image("Images/image (1).jpg")

st.title(" ")

persona = """   I’m Rohan, a Mechanical Engineer with a specialization in Mechatronics, and I’ve transitioned into Operations Management at Amazon. With over 3 years of experience in research, data analytics, operations management, and project leadership, I thrive on cross-functional leadership, process optimization, and innovative problem-solving.
                In my current role as a Manager at Amazon since 2022, I lead operations in an automated fulfillment center, managing a team of over 100 associates and driving operational excellence. I’ve honed my expertise in troubleshooting, commissioning, and optimizing AI-based systems, achieving a consistent 95%+ on-time metric, reducing operational errors by 15%, and improving efficiency by 20%. Additionally, I led a green dunnage airbags recycling project that resulted in $88,000 in annual savings and a reduction of 1.2 tons of carbon emissions. My work is rooted in fostering an inclusive and supportive environment, particularly for immigrant communities, which has led to a 25% increase in employee engagement and retention.
                Prior to Amazon, I had enriching experiences as a Mechatronics Engineering Intern at Hover City Inc. and an Innovation and Business Development Intern at Toronto Hydro. At Hover City, I contributed to the design of a modular water system for a fully autonomous flying home. I led the research, 3D CAD modeling, and Finite Element Analysis (FEA), which reduced manufacturing time by 12% and scrap production by 32%. Collaborating with aerodynamics and systems engineering teams, I integrated the water system seamlessly into the vehicle structure, improving performance by 25%.
                At Toronto Hydro, I played a key role in launching five next-generation EV charging pilot projects in the City of Toronto. I used Alteryx and SQL to analyze charging data, improving utilization by 40%. My involvement in community roadshows helped drive a 40% increase in customer inquiries and a 25% growth in market share for EV infrastructure.
                Academically, I hold a Bachelor of Mechanical Engineering from Toronto Metropolitan University (formerly Ryerson University) with a specialization in Mechatronics in 2021. I was recognized on the Dean’s List and earned an A+ in my Capstone Design Project. I was also a finalist in the Boeing Go Fly Competition, where I showcased an eVTOL aircraft prototype.
                Throughout my career, I’ve developed proficiency in a wide array of tools and platforms, including Microsoft Office Suite, Project, Visio, JIRA, SolidWorks, ANSYS, AutoCAD, Tableau, Alteryx, Python, and SQL. I’ve led several technical projects, such as developing an AI-powered object detection tool using the YOLOv8 algorithm, and a breast cancer classifier using machine learning.
                I am passionate about entrepreneurship, sustainability, and innovation.My diverse experience across operations, technical design, and product development drives my goal to continue growing in fields like AI, robotics, and sustainable technology. You can connect with me via linkedin: www.linkedin.com/in/rd-rohan-doshi. You can also email me at rohannavindoshi@gmail.com. You can call: +1647-982-0448. I am currently located in Vancouver, Canada. Answer all the questions in first person view and if not sure about any of response respond "Its a secret"  """

st.title("Chat with virtual me")
user_question = st.text_input("Ask anything you would like to know about me?")   #detects input


if st.button("ASK",use_container_width=400):
    prompt = persona + user_question
    response = model.generate_content(prompt)
    st.write(response.text)

st.title("Projects Gallery")

col5,col6 = st.columns([4,4])
with col5:
    st.subheader("Interactive Gesture Control Map")
    st.video("Videos/Interactive_Map.mp4")
with col6:
    st.subheader("Autonomous Robotic Vehicle")
    st.video("Videos/Robotic_Vehicle.mov")

st.subheader("Interactive AI Content Generator ")
st.text_input("This project is an content generator powered by large language generator trained by Google GeminiAI that enables users to draw with their fingers in air as input and generated dynamic and personalized responses")
st.image("Images/HandSign.jpg", width = 350)
st.subheader("Lets try it yourself")


col3 , col4= st.columns([2,1])
with col3:
    run = st.checkbox('Run', value=True)
    FRAME_WINDOW = st.image([])

with col4:
    st.title ("Answer")
    output_text_area = st.subheader("")



from cvzone.HandTrackingModule import HandDetector

    # Initialize the webcam to capture video
    # The '2' indicates the third camera connected to your computer; '0' would usually refer to the built-in camera
cap = cv2.VideoCapture(0)
cap.set(2,480)
cap.set(2,480)


# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)

def getHandInfo(img):
    hands, img = detector.findHands(img, draw=True, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    else:
        return None


def draw (info, prev_pos,canvas):
    fingers, lmlist = info
    current_pos = None
    if fingers ==[0,1,0,0,0]:
        current_pos = lmlist [8][0:2]
        if prev_pos is None: prev_pos = current_pos
        cv2.line(canvas,current_pos,prev_pos, (255,0,255), 10)

    elif fingers == [1, 1, 0, 0, 1]:  # All fingers up (reset canvas)
        canvas = np.zeros_like(canvas)  # Correctly reset canvas

    return current_pos, canvas
    return current_pos, canvas

def sendToAI(model, canvas,fingers):
    if fingers == [1,1,1,1,1]:
        pil_image = Image.fromarray (canvas)
        response = model.generate_content(["guess the answer.", pil_image])
        return response.text

prev_pos = None
canvas = None
image_combined = None
output_text = " "

# Continuously get frames from the webcam
while True:
    # Capture each frame from the webcam
    # 'success' will be True if the frame is successfully captured, 'img' will contain the frame
    success, img = cap.read()
    img = cv2.flip(img, flipCode=1)

    if canvas is None:
        canvas = np.zeros_like(img)


    info = getHandInfo(img)
    if info:
        fingers, lmlist = info
        prev_pos, canvas = draw(info,prev_pos,canvas)
        output_text = sendToAI(model, canvas, fingers)
    image_combined = cv2.addWeighted(img, 0.80, canvas, 0.20, 0)
    FRAME_WINDOW.image(image_combined, channels="BGR")

    if output_text:
        output_text_area.text(output_text)

    #cv2.imshow("Image", img)
    #cv2.imshow("Canvas", canvas)
    #cv2.imshow("image_combined", image_combined)

    # Keep the window open and update it for each frame; wait for 1 millisecond between frames
    cv2.waitKey(1)

