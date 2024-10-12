# Importing the required dependencies
import cv2                                     # for video rendering
import dlib                                    # for face and landmark detection
import imutils
from imutils import face_utils                 # for face utilities
import time

rtsp_url = "rtsp://admin:123456789tung@192.168.0.110:554/ch1/main"
rtmp_url = "rtmp://62.146.236.233/live/Ancam1"
# Initialize webcam (0 is the default webcam)
cam = cv2.VideoCapture(rtsp_url)

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

    # Giữ nguyên kích thước để xử lý
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

        # Calculate the center of the bounding box
        center_x = x + w // 2
        center_y = y + h // 2
        cv2.circle(img, (center_x, center_y), 5,
                   (0, 0, 255), -1)  # Vẽ điểm chính giữa

        # Landmark detection
        shape = landmark_predict(img_gray, face)
        shape = face_utils.shape_to_np(shape)
        # Xóa hoặc bình luận lại đoạn mã vẽ chấm tròn
        # for lm in shape:
        #     cv2.circle(img, tuple(lm), 3, (10, 2, 200))

    # Display face count
    cv2.putText(img, f'Faces: {face_count}', (30, 70),
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
    cv2.imshow("webcam_face_detection", img)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cam.release()
cv2.destroyAllWindows()
