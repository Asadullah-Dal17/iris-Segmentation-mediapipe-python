import cv2 as cv
import numpy as np
import mediapipe as mp
import math
mp_face_mesh = mp.solutions.face_mesh

LEFT_IRIS = [474,475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ] 

cap = cv.VideoCapture('eyes_movement.mp4')
# Euclaidean distance 
def euclidean_distance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

def position_indicator(img,dist1, total_dist, eye='R-eye: ',text_pos = (30,40),):
    ratio1 = dist1/total_dist
    pos = ""
    color=(0,0,0)
    if ratio1 <=0.44:
        pos = "left"
        color=(0,255,255)
    elif ratio1 >=0.56:
        pos ='right'
        color=(0,255,0)
    else: 
        pos = 'center'
        color=(0,0,255)
  
    cv.putText(img, f"{eye} {pos} {round(ratio1,3)}", text_pos, cv.FONT_HERSHEY_PLAIN, 1.2, color, 2, cv.LINE_AA)

with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while True:
        ret, frame = cap.read()
        frame =cv.flip(frame, 1)    
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        img_h, img_w = frame.shape[:2]
        if results.multi_face_landmarks:
            # print(results.multi_face_landmarks[0].landmark)
            face_mesh_point =np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
            # print(face_mesh_point.shape)
            # cv.polylines(frame, [face_mesh_point[LEFT_IRIS]], True, (0,255,0), 2)
            (cx_l, cy_l), radius = cv.minEnclosingCircle(face_mesh_point[LEFT_IRIS])
            center_left = np.array([cx_l, cy_l], dtype = np.int32)
            (cx_r, cy_r), radius = cv.minEnclosingCircle(face_mesh_point[RIGHT_IRIS])
            center_right = np.array([cx_r, cy_r], dtype = np.int32)
            cv.circle(frame, center_right, int(radius), (0,255,255), 1, cv.LINE_AA)
            cv.circle(frame, center_left, int(radius), (0,255,255), 1, cv.LINE_AA)
            cv.circle(frame, center_right, 2, (0,255,255), -1, cv.LINE_AA)
            cv.circle(frame, center_left, 2, (0,255,255), -1, cv.LINE_AA)
            cv.circle(frame, face_mesh_point[LEFT_EYE[8]], 3, (0,255,255), -1, cv.LINE_AA)
            cv.circle(frame, face_mesh_point[RIGHT_EYE[8]], 3, (0,255,255), -1, cv.LINE_AA)
            cv.circle(frame, face_mesh_point[LEFT_EYE[0]], 3, (0,255,0), -1, cv.LINE_AA)
            cv.circle(frame, face_mesh_point[RIGHT_EYE[0]], 3, (0,255,0), -1, cv.LINE_AA)
            hr_dist_left = euclidean_distance(face_mesh_point[LEFT_EYE[0]], face_mesh_point[LEFT_EYE[8]])
            hr_dist_right = euclidean_distance(face_mesh_point[RIGHT_EYE[0]], face_mesh_point[RIGHT_EYE[8]])
            center_right_dist = euclidean_distance(face_mesh_point[RIGHT_EYE[0]], center_right)
            center_left_dist = euclidean_distance(face_mesh_point[LEFT_EYE[0]], center_left)
            position_indicator(frame, center_right_dist, hr_dist_right)
            position_indicator(frame, center_left_dist, hr_dist_left,eye="L-eye: " ,text_pos=(30,70))
        cv.imshow('img', frame)
        key = cv.waitKey(1)
        if key ==ord('q'):
            break
cv.destroyAllWindows()
cap.release()