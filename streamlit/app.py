import tempfile
import cv2
import streamlit as st
import numpy as np
import mediapipe as mp
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import requests

post_url = 'http://127.0.0.1:8000/prediction'

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils 

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False                 
    results = model.process(image)                 
    image.flags.writeable = True                  
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks,
                              mp_holistic.FACEMESH_TESSELATION) 
    
    mp_drawing.draw_landmarks(image, results.pose_landmarks,
                              mp_holistic.POSE_CONNECTIONS) 
    
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS) 
    
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS) 
    
def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks,
                              mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10),
                                                    thickness=1,
                                                    circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121),
                                                    thickness=1,
                                                    circle_radius=1)
                             ) 
    
    mp_drawing.draw_landmarks(image, results.pose_landmarks,
                              mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10),
                                                    thickness=2,
                                                    circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121),
                                                    thickness=2,
                                                    circle_radius=2)
                             ) 
    
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76),
                                                    thickness=2,
                                                    circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250),
                                                    thickness=2,
                                                    circle_radius=2)
                             )
    
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66),
                                                    thickness=2,
                                                    circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230),
                                                    thickness=2,
                                                    circle_radius=2)
                             ) 

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    #print(pose)

    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    #print(face)

    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    #print(left_hand)

    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    #print(right_hand)
    
    return np.concatenate([pose, face, left_hand, right_hand])

f = st.file_uploader("Choose a Video")   
if f:   
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(f.read())
    list_data = []   
    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()

    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(total_frames)

    with mp_holistic.Holistic(min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
    
        for frame_num in range(int(total_frames)):

            ret, frame = cap.read()
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            keypoints = extract_keypoints(results)
            list_data.append(keypoints)
        
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    arr_data = np.array(list_data)
    arr_before_normalize = arr_data
    st.video(tfile.name)

    if st.button('Predict Data'):
        data = {'keypionts': arr_before_normalize.tolist()}
        x = requests.post(post_url, json = data)
        # st.write(x.text)
        st.write('Model 1 : มึนศีรษะ')
        st.write('Model 2 : มึนศีรษะ')
        

else:
    pass