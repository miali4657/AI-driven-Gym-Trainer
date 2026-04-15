import streamlit as st
import mediapipe as mp
import numpy as np
import time
import cv2
import os

# Suppress MediaPipe warnings
os.environ['GLOG_minloglevel'] = '2'

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# App title and sidebar
st.title('Smart Exercise using MediaPipe')
st.sidebar.title('Smart Exercise using MediaPipe and Streamlit')

app_mode = st.sidebar.selectbox('Select Mode', ['Training', 'About App'])

if app_mode == 'About App':
    st.markdown("""
    ## About This App
    This application uses MediaPipe for detecting exercise movements and provides real-time feedback.
    Currently supports:
    - Side Arises
    - Standing Curls
    - Squats
    """)
    # Add more about info as needed

elif app_mode == 'Training':
    # Initialize session state for exercise selection and counter
    if 'exercise' not in st.session_state:
        st.session_state.exercise = None
    if 'counter' not in st.session_state:
        st.session_state.counter = 0
    if 'stage' not in st.session_state:
        st.session_state.stage = None

    # Exercise selection buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Side Arises"):
            st.session_state.exercise = "side_arises"
            st.session_state.counter = 0
    with col2:
        if st.button("Standing Curls"):
            st.session_state.exercise = "standing_curls"
            st.session_state.counter = 0
    with col3:
        if st.button("Squats"):
            st.session_state.exercise = "squats"
            st.session_state.counter = 0

    # Repetition selector
    target_reps = st.select_slider(
        'Select target repetitions',
        options=['5', '10', '15', '20', '25', '30'],
        value='10'
    )

    # Confidence settings
    detection_confidence = st.sidebar.slider('Min Detection Confidence', 0.0, 1.0, 0.5)
    tracking_confidence = st.sidebar.slider('Min Tracking Confidence', 0.0, 1.0, 0.5)

    # Video feed
    stframe = st.empty()
    kpi1, kpi2, kpi3 = st.columns(3)
    
    with kpi1:
        st.markdown("**Frame Rate**")
        kpi1_text = st.markdown("0")
    
    with kpi2:
        st.markdown("**Count**")
        kpi2_text = st.markdown("0")
    
    with kpi3:
        st.markdown("**Target**")
        kpi3_text = st.markdown(target_reps)

    # Helper functions
    def calculate_angle(a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
            
        return angle

    # Initialize video capture
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 0
    prev_time = 0

    with mp_pose.Pose(
        min_detection_confidence=detection_confidence,
        min_tracking_confidence=tracking_confidence
    ) as pose:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            # Convert to RGB and process with MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get key points
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                # Calculate angles
                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                knee_angle = calculate_angle(hip, knee, ankle)
                shoulder_angle = calculate_angle(elbow, shoulder, hip)

                # Exercise detection logic
                if st.session_state.exercise == "squats":
                    if knee_angle > 160:
                        st.session_state.stage = "up"
                    if knee_angle < 90 and st.session_state.stage == "up":
                        st.session_state.stage = "down"
                        st.session_state.counter += 1
                        
                elif st.session_state.exercise == "standing_curls":
                    if elbow_angle > 160:
                        st.session_state.stage = "down"
                    if elbow_angle < 50 and st.session_state.stage == "down":
                        st.session_state.stage = "up"
                        st.session_state.counter += 1
                        
                elif st.session_state.exercise == "side_arises":
                    if shoulder_angle > 40:
                        st.session_state.stage = "up"
                    if shoulder_angle < 20 and st.session_state.stage == "up":
                        st.session_state.stage = "down"
                        st.session_state.counter += 1

            except:
                pass

            # Draw landmarks
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Display counter
            cv2.putText(image, f'Count: {st.session_state.counter}', 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            # Update display
            kpi1_text.write(f"<h1 style='text-align: center; color: red;'>{int(fps)}</h1>", 
                           unsafe_allow_html=True)
            kpi2_text.write(f"<h1 style='text-align: center; color: red;'>{st.session_state.counter}</h1>", 
                           unsafe_allow_html=True)
            kpi3_text.write(f"<h1 style='text-align: center; color: red;'>{target_reps}</h1>", 
                           unsafe_allow_html=True)
            
            # Show the frame
            stframe.image(image, channels='RGB', use_column_width=True)

            # Exit if target reached
            if st.session_state.counter >= int(target_reps):
                st.balloons()
                st.success("Target reached! Great job!")
                st.session_state.exercise = None
                break

    cap.release()