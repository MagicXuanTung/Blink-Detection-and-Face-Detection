# Importing the required dependencies
import cv2                                     # for video rendering
import dlib                                    # for face and landmark detection
import imutils
# for calculating distance between the eye landmarks
from scipy.spatial import distance as dist
# to get the landmark ids of the left and right eyes
from imutils import face_utils
import time

# Initialize webcam (0 is the default webcam)
cam = cv2.VideoCapture(0)

# Define a function to calculate the EAR (Eye Aspect Ratio)


def calculate_EAR(eye):
    # Calculate the vertical distances
    y1 = dist.euclidean(eye[1], eye[5])
    y2 = dist.euclidean(eye[2], eye[4])

    # Calculate the horizontal distance
    x1 = dist.euclidean(eye[0], eye[3])

    # Calculate the EAR (Eye Aspect Ratio)
    EAR = (y1 + y2) / x1
    return EAR

# Mark the eye landmarks on the image


def mark_eyeLandmark(img, eyes):
    for eye in eyes:
        pt1, pt2 = (eye[1], eye[5])
        pt3, pt4 = (eye[0], eye[3])
        cv2.line(img, pt1, pt2, (200, 0, 0), 2)
        cv2.line(img, pt3, pt4, (200, 0, 0), 2)
    return img


# Variables
blink_thresh = 0.5
succ_frame = 2
count_frame = 0
blink_count = 0

# Eye landmarks indices
(L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

# Initialize the Models for Landmark and Face Detection
detector = dlib.get_frontal_face_detector()
landmark_predict = dlib.shape_predictor(
    'Model/shape_predictor_68_face_landmarks.dat')

# Initialize FPS calculation variables
fps_start_time = 0
fps = 0
frame_count = 0

while True:
    # Capture frame-by-frame
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = imutils.resize(frame, width=640)
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(img_gray)
    img = frame.copy()  # Ensure img is initialized here

    # Count the number of faces detected
    face_count = len(faces)

    for face in faces:
        # Draw a bounding box around the face
        (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Landmark detection
        shape = landmark_predict(img_gray, face)
        shape = face_utils.shape_to_np(shape)
        for lm in shape:
            cv2.circle(img, tuple(lm), 3, (10, 2, 200))

        # Extract left and right eye landmarks
        lefteye = shape[L_start:L_end]
        righteye = shape[R_start:R_end]

        # Calculate EAR for both eyes
        left_EAR = calculate_EAR(lefteye)
        right_EAR = calculate_EAR(righteye)
        img = mark_eyeLandmark(img, [lefteye, righteye])

        # Average EAR of both eyes
        avg = (left_EAR + right_EAR) / 2
        if avg < blink_thresh:
            count_frame += 1
        else:
            if count_frame >= succ_frame:
                blink_count += 1
                cv2.putText(img, 'Blink Detected', (30, 30),
                            cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)
            count_frame = 0

    # Display blink count
    cv2.putText(img, f'Blinks: {blink_count}', (30, 70),
                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)

    # Display face count
    cv2.putText(img, f'Faces: {face_count}', (30, 150),
                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)

    # Calculate FPS
    frame_count += 1
    if frame_count == 1:
        fps_start_time = time.time()
    else:
        fps_end_time = time.time()
        fps = frame_count / (fps_end_time - fps_start_time)
        cv2.putText(img, f'FPS: {int(fps)}', (30, 110),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)

    # Display the resulting frame
    cv2.imshow("webcam_full_option", img)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cam.release()
cv2.destroyAllWindows()
