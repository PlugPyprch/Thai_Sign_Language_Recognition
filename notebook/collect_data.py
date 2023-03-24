import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

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
    
def ex_key(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    #print(pose)

    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    #print(face)

    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    #print(left_hand)

    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    #print(right_hand)
    
    return np.concatenate([pose, face, left_hand, right_hand])

def record():
    class_name = input('Class Name : ')
    class_a = ''
    
    f = open("class_names.txt", "r")
    text = f.read()
    f.close()
    text = text[1:]
    x = text.split(",")
    for i, b in enumerate(x):
        if class_name == b:
            class_a = str(i+1)
    # print(class_a)
    
    if not class_name in x:
        print('')
        print('please add class before recording!!')
        print('')
        return None
    
    ex = 1
    class_type = 1
    cap= cv2.VideoCapture(0)
    width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    number = 0
    rec_o = 0
    
    while True:
        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, 'Class : {}'.format(class_name), (15, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow('OpenCV Feed', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('e'):
                ex = 0
                break
            elif key == ord('a'):
                rec_o = 1
                name = 1
                number = name
                dir = os.listdir(os.path.join('videos', class_name))
                if len(dir) == 0:
                    #print("Empty directory")
                    writer= cv2.VideoWriter(r'videos/'+ class_name +'/{}_{}.mp4'.format(class_a, name), cv2.VideoWriter_fourcc(*'mp4v'), 30, (width,height))
                else:
                    #print("Not empty directory")
                    for f in os.listdir(os.path.join('videos', class_name)):
                        if not f.startswith('.'):
                            if int(f[2:-4]) >= name:
                                name = int(f[2:-4]) + 1
                                number = name
                    writer= cv2.VideoWriter(r'videos/'+ class_name +'/{}_{}.mp4'.format(class_a, name), cv2.VideoWriter_fourcc(*'mp4v'), 30, (width,height))
                break
            

        if ex == 1:
            cap= cv2.VideoCapture(0)
            with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                for j in range(90):

                    ret, frame = cap.read()
                    frame = cv2.flip(frame, 1)
                    writer.write(frame)

                    frame, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(frame, results)
                    
                    cv2.putText(frame, 'Class : {}'.format(class_name), (15, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                    
                    cv2.putText(frame, 'Video No. : {}'.format(number), (15, 190),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

                    if j <= 30:
                        if class_type == 1:
                            cv2.putText(frame, 'Collecting : {}'.format(3), (15, 120),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
                            cv2.imshow('OpenCV Feed', frame)
                        else:
                            cv2.putText(frame, 'Collecting : {}'.format(3), (15, 120),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                            cv2.imshow('OpenCV Feed', frame)

                    elif j > 30 and j <=60:
                        if class_type == 1:
                            cv2.putText(frame, 'Collecting : {}'.format(2), (15, 120),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
                            cv2.imshow('OpenCV Feed', frame)
                        else:
                            cv2.putText(frame, 'Collecting : {}'.format(2), (15, 120),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                            cv2.imshow('OpenCV Feed', frame)

                    elif j >60:
                        if class_type == 1:
                            cv2.putText(frame, 'Collecting : {}'.format(1), (15, 120),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
                            cv2.imshow('OpenCV Feed', frame)    
                        else:
                            cv2.putText(frame, 'Collecting : {}'.format(1), (15, 120),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                            cv2.imshow('OpenCV Feed', frame)   

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break     
        else:
            break
    
    if rec_o == 1:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()
    for i in range (1,5):
        cv2.waitKey(1)
        
def extract_keypoints():
    f = open("class_names.txt", "r")
    text = f.read()
    f.close()
    text = text[1:]
    class_name = text.split(",")
    
    for i, b in enumerate(class_name):
        dir = []
        for f in os.listdir(os.path.join('keypoints', b)):
                if not f.startswith('.'):
                    dir.append(f)
        if len(dir) == 0:
            #print('folder empty')
            for f in os.listdir(os.path.join('videos', b)):
                if not f.startswith('.'):
                    os.makedirs(os.path.join('keypoints', b, f[2:-4]))
                    cap = cv2.VideoCapture(os.path.join('videos', b, f))
                    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                        # Loop through video length aka sequence length
                        for frame_num in range(90):

                            # Read feed
                            ret, frame = cap.read()

                            # Make detections
                            image, results = mediapipe_detection(frame, holistic)

                            # Draw landmarks
                            draw_styled_landmarks(image, results)

                            
                            if frame_num == 0: 
                                
                                cv2.putText(image, 'file_name : {} || class : {}'.format(f, b), (15,50), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                                
                                cv2.imshow('OpenCV Feed', image)
                                
                            else: 
                                cv2.putText(image, 'file_name : {} || class : {}'.format(f, b), (15,50), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                              
                                cv2.imshow('OpenCV Feed', image)

                            # NEW Export keypoints
                            keypoints = ex_key(results)
                            npy_path = os.path.join('keypoints', b, f[2:-4], str(frame_num))
                            np.save(npy_path, keypoints)

                            # Break gracefully
                            if cv2.waitKey(10) & 0xFF == ord('q'):
                                break

                    cap.release()
                    cv2.destroyAllWindows()
                    for i in range (1,5):
                        cv2.waitKey(1)
        else:
            #print('Folder not empty')
            dir_videos = []
            for f in os.listdir(os.path.join('videos', b)):
                    if not f.startswith('.'):
                        dir_videos.append(f)
                        
            if len(dir) == len(dir_videos):
                print('No new file recorded')
            
            elif len(dir_videos) > len(dir):
                max_videos = -1
                max_keypoints = -1
                
                for f in os.listdir(os.path.join('videos', b)):
                    if not f.startswith('.'):
                        if (int(f[2:-4])) > max_videos:
                            max_videos = int(f[2:-4])
                
                for f in os.listdir(os.path.join('keypoints', b)):
                    if not f.startswith('.'):
                        if (int(f)) > max_keypoints:
                            max_keypoints = int(f)
                
                name = max_keypoints+1
                
                while name <= max_videos:
                    
                    os.makedirs(os.path.join('keypoints', b, str(name)))
                    f = open("class_names.txt", "r")
                    text = f.read()
                    f.close()
                    text = text[1:]
                    x = text.split(",")
                    class_a = ''
                    for i, w in enumerate(x):
                        if b == w:
                            class_a = str(i+1)
                    
                    f = '{}_{}.mp4'.format(class_a, str(name))
                    
                    cap = cv2.VideoCapture(os.path.join('videos', b, f))
                    
                    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                        
                        # Loop through video length aka sequence length
                        for frame_num in range(90):

                            # Read feed
                            ret, frame = cap.read()

                            # Make detections
                            image, results = mediapipe_detection(frame, holistic)

                            # Draw landmarks
                            draw_styled_landmarks(image, results)

                            
                            if frame_num == 0: 
                                
                                cv2.putText(image, 'file_name : {} || class : {}'.format(f, b), (15,50), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                                
                                cv2.imshow('OpenCV Feed', image)
                                
                            else: 
                                cv2.putText(image, 'file_name : {} || class : {}'.format(f, b), (15,50), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                              
                                cv2.imshow('OpenCV Feed', image)

                            # NEW Export keypoints
                            keypoints = ex_key(results)
                            npy_path = os.path.join('keypoints', b, str(name), str(frame_num))
                            np.save(npy_path, keypoints)

                            # Break gracefully
                            if cv2.waitKey(10) & 0xFF == ord('q'):
                                break

                    cap.release()
                    name = name + 1
                    cv2.destroyAllWindows()
                    for i in range (1,5):
                        cv2.waitKey(1)
                    

def setupFolder():
    os.makedirs(os.path.join('keypoints'))
    os.makedirs(os.path.join('videos'))
    f = open("class_names.txt", "x")
    
def print_class_name():
    f = open("class_names.txt", "r")
    text = f.read()
    f.close()
    text = text[1:]
    x = text.split(",")
    print(x)
    
def add_class():
    new_class = input('New Class Name : ')
    f = open("class_names.txt", "a") 
    f.write(',' + new_class)    
    f.close()
    os.makedirs(os.path.join('keypoints', new_class))
    os.makedirs(os.path.join('videos', new_class))

def check_emp():
    try:
        dir_a = os.listdir(os.path.join('keypoints'))
    except:
        #print("An exception occurred")
        return 'new'

def main():
    i = 'loop'
    while i=='loop':
        cv2.destroyAllWindows()
        print('')
        print('<--------------------------Sign Language Data Collector-------------------------------->')
        print('')
        print('** input function number for use function')
        print('')
        print(r'> [1] setupFolder')
        print(r'> [2] print_class_name')
        print(r'> [3] add_class')
        print(r'> [4] record')
        print(r'> [5] extract_keypoints')
        print(r'> [0] exit')
        print('')
        print('** for [4] record function : press(a) to record video and press(e) to close window')
        print('** this program recommend for collect sign landguage gesture only 1-9 classes')
        print('')
        print('<-------------------------------------------------------------------------------------->')
        print('')
        x = input('Select Function : ')
        print('')
        
        if x == '0':
            break
            
        elif x == '1':
            setupFolder()
            print('')
            print('Create Folder Success!')
            print('')
            
        elif x == '2':
            a = check_emp()
            if a == 'new':
                print('Please <setupFolder> before do anythings!!')
            else:
                print_class_name()
            
        elif x == '3':
            a = check_emp()
            if a == 'new':
                print('Please <setupFolder> before do anythings!!')
            else:
                add_class()
            
        elif x == '4':
            a = check_emp()
            if a == 'new':
                print('Please <setupFolder> before do anythings!!')
            else:
                record()
            
        elif x == '5':
            a = check_emp()
            if a == 'new':
                print('Please <setupFolder> before do anythings!!')
            else:
                extract_keypoints()
            
        else:
            print('Wrong Input!!')
            

if __name__ == "__main__":
    main()
                
    