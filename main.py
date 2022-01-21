
import cv2 as cv
import numpy as np
import mediapipe as mp
import numpy as np
import math


# Euclaidean distance 
def euclideanDistance(point, point1):
    x, y = point
    # print(x, y)
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

def indicator(img,dist1, total_dist, eye="R-eye:", text_pos=(30,70)):
    ratio1 = dist1/total_dist
    pos = ""
    if ratio1 <=0.44:
        pos = "left"
        color =(0,0,255)
    elif ratio1 >=0.55:
        pos ='right'
        color =(0,255,255)
    else: 
        pos = 'center'
        color =(0,255,0)
  
    cv.putText(img, f"{eye} Looking {pos} {round(ratio1,3)}", text_pos, cv.FONT_HERSHEY_PLAIN, 1.2, color, 2, cv.LINE_AA)

mp_face_mesh = mp.solutions.face_mesh


LEFT_IRIS = [474,475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ] 
cap = cv.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces=1, 
    refine_landmarks=True, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            # print(results.multi_face_landmarks[0].landmark)
            mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) 
            for p in results.multi_face_landmarks[0].landmark])
            # print(mesh_points)
            # cv.polylines(frame, [mesh_points[LEFT_IRIS]],True, (0,255,0), 1)
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
            left_center = np.array([l_cx, l_cy], dtype= np.int32)
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            right_center = np.array([r_cx, r_cy], dtype= np.int32)
            cv.circle(frame, left_center, int(l_radius), (0,255,0), 1, cv.LINE_AA)
            cv.circle(frame, left_center, 1, (0,255,255), -1, cv.LINE_AA)
            cv.circle(frame, right_center, int(r_radius), (0,255,0), 1, cv.LINE_AA)
            cv.circle(frame, right_center, 1, (0,255,255), -1, cv.LINE_AA)

            # cv.circle(frame, mesh_points[LEFT_EYE[0]], 2, (0,255,0), -1, cv.LINE_AA)
            # cv.circle(frame, mesh_points[LEFT_EYE[8]], 2, (0,255,255), -1, cv.LINE_AA)
            right_dist = euclideanDistance(mesh_points[RIGHT_EYE[0]], right_center)
            right_dist_total = euclideanDistance(mesh_points[RIGHT_EYE[0]], mesh_points[RIGHT_EYE[8]])
            indicator(frame, right_dist, right_dist_total)
            left_dist = euclideanDistance(mesh_points[LEFT_EYE[0]], left_center)
            left_dist_total = euclideanDistance(mesh_points[LEFT_EYE[0]], mesh_points[LEFT_EYE[8]])
            indicator(frame, left_dist, left_dist_total, eye="L_eye:", text_pos=(30,40))

        cv.imshow("img", frame)
        key = cv.waitKey(1)
        if key ==ord('q'):
            break
cap.release()
cv.destroyAllWindows()