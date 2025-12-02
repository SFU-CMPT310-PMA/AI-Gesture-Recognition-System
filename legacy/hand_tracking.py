import os
import csv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time
from mediapipe.framework.formats import landmark_pb2
import tensorflow as tf
import numpy as np
from backend.sign_detection import toLabelHandSigns

# Draw landmarks on the frame
def draw_landmarks(frame, landmarks):
    for lm in landmarks:
        h, w, _ = frame.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

def draw_label(frame, label_text, hand_landmarks):
    h, w, _ = frame.shape
    # find the landmark that's lowest on the screen
    lowest_landmark = max(hand_landmarks,  key = lambda lm: lm.y)
    x_pixel = int(lowest_landmark.x * w)
    y_pixel = int(lowest_landmark.y * h)
    box_width, box_height = 120, 30
    overlay = frame.copy()

    top_left_x = x_pixel - box_width // 2
    bottom_right_x = x_pixel + box_width // 2
    bottom_right_y = y_pixel + box_height

    cv2.rectangle(
        overlay,
        (top_left_x, y_pixel),
        (bottom_right_x, bottom_right_y),
        (0, 0, 0),
        -1  # thcikness, this indicates a solid fill
    )
    transparentAmt = 0.6
    cv2.addWeighted(overlay, transparentAmt, frame, 1 - transparentAmt, 0, frame)
    cv2.putText(
        frame,
        label_text,
        (top_left_x + 5, y_pixel + 22),  # kinda offseted
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,  # font size
        (255, 255, 255),  # text colour white
        2  # thickness
    )

def print_result(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global hand_landmarker_result
    hand_landmarker_result = result
    

# Save landmarks for the dataset
def save_landmarks_to_csv(label, landmarks):
    file_path = "hand_gesture_dataset.csv"
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write header if file is new
        if not file_exists:
            header = ["label"]
            for i in range(21):
                header += [f"x{i+1}", f"y{i+1}", f"z{i+1}"]
                writer.writerow(header)
        row = [label]
        for landmark in landmarks:
            row.extend([landmark.x, landmark.y, landmark.z])
        writer.writerow(row)


# Get feature vectors from the webcam
def getXFeatures(handlandmark_results):
    X = np.array([[landmark.x, landmark.y, landmark.z] for landmark in handlandmark_results]).flatten()
    return X.reshape(1, -1)

def predictLabels(model, X):
    y_pred_distribution = model.predict(X, verbose = 0)
    y_pred = np.argmax(y_pred_distribution, axis= 1)
    y_pred_label = toLabelHandSigns(y_pred)
    print(y_pred_label)
    return y_pred_label

# https://stackoverflow.com/questions/14063070/overlay-a-smaller-image-on-a-larger-image-python-opencv
# https://stackoverflow.com/questions/32290096/python-opencv-add-alpha-channel-to-rgb-image/32290192#32290192
def window_border(img, img_overlay, pos = (0, 0), alpha_mask = None):
    x, y = pos
    ## fit background
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])
    ## overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)
    max_rgb = 255.0
    
    if alpha_mask is None:
        alpha = (img_overlay[y1o:y2o, x1o:x2o, 3] / max_rgb)
    else:
        alpha = (alpha_mask[y1o:y2o, x1o:x2o] / max_rgb)

    for colour in range(3): 
        img[y1:y2, x1:x2, colour] = (1 - alpha) * img[y1:y2, x1:x2, colour] + alpha * img_overlay[y1o:y2o, x1o:x2o, colour]


def main():
    """
    1. Initialize and return a MediaPipe Hand Landmarker detector
    """
    ## Store model path for the Hand Landmark task ##
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "model", "hand_landmarker.task")

    # Basic configuration for any MediaPipe task
    BaseOptions = python.BaseOptions

    # Class detect hand landmarks                   
    HandLandmarker = vision.HandLandmarker

    # Configuration for the hand landmark
    HandLandmarkerOptions = vision.HandLandmarkerOptions

    # Define how the model run
    VisionRunningMode = vision.RunningMode

    ## Create a hand landmarker instance with the real-time webcam ##
    options = HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path),
                                        running_mode=VisionRunningMode.LIVE_STREAM, result_callback = print_result) 
    detector = HandLandmarker.create_from_options(options)


    """
    2. Initialize and Run OpenCV to Open Webcam
    """
    # Create a VideoCapture object and Use the default camera
    cam = cv2.VideoCapture(0)

    # Check if the camera actually opened succesfully
    if not cam.isOpened():
        print("Cannot open camera. Exiting...")
        exit()
    overlay_img = cv2.imread("images/overlay.png", cv2.IMREAD_UNCHANGED)

    print("\n=== DATA COLLECTION MODE ===")
    print("Press 'r' for Rock ✊, 'p' for Paper 🖐️, 's' for Scissors ✌️")
    print("Press 'q' to quit\n")

    while True:
        # Read a single frame from the camera
        # ret = Return value -> True if frame read succesfully, else return False
        # frame = a NumPy array representing the current image (h x w x 3 color channels)
        ret, frame = cam.read()

        # If reading fails, stop the loop
        if not ret: 
            print("Can't receive frame. Exiting....")
            break

        # Mirror the frame
        frame = cv2.flip(frame, 1)
        if overlay_img is not None:
            overlay_resized = cv2.resize(overlay_img, (frame.shape[1], frame.shape[0]), interpolation = cv2.INTER_AREA)
            window_border(frame, overlay_resized, pos=(0, 0))

        """
        3. Prepare frames from OpenCV and process it for MediaPipe
        """
        # Prepare data
        mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data=frame)

        """
        4. Run detectection and Show the handlandmark result
        """
        # Detect hand landmarks from the input real-time webcam
        frame_timestamp = (int) (time.time() * 1000) #in millisecond
        detector.detect_async(mp_image, frame_timestamp)


        """
        5. Visualize the result
        """
        # If our hand landmark results aren't valid skip since nothing to visualize
        if hand_landmarker_result and hand_landmarker_result.hand_landmarks:
            # Note: you may see some informational startup logs at the beginning.
            for hand_landmarks in hand_landmarker_result.hand_landmarks:
                # Draw the hand landmarks using x,y,z coordinates
                # Note : Must be in NormalizedLandmarkList format
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                for landmark in hand_landmarks:
                    landmark = landmark_pb2.NormalizedLandmark(x = landmark.x, y = landmark.y, z = landmark.z)
                    hand_landmarks_proto.landmark.append(landmark)

                # Draw the landmarks on the OpenCV frame
                draw_landmark.draw_landmarks(
                    image = frame,
                    landmark_list = hand_landmarks_proto,
                    connections = mp_hands.HAND_CONNECTIONS, 
                    # landmark points specs
                    landmark_drawing_spec=draw_landmark.DrawingSpec( 
                        color = (0, 153, 76),   # colours for landmarks, uses BGR 
                        thickness = 2,         # line thickness for points
                        circle_radius = 2),    # each points radius
                    # landmark connection lines specs
                    connection_drawing_spec=draw_landmark.DrawingSpec(
                        color = (102, 255, 178), # connection color, uses BGR
                        thickness = 2)         # line thickness for connection
                )
                """
                5. TODO: Add a label that shows either rock/paper/scissors/unknown
                """
        
        if hand_landmarker_result and hand_landmarker_result.hand_landmarks:
            # Draw landmarks
            draw_landmarks(frame, hand_landmarker_result.hand_landmarks[0])
            draw_label(frame, "unknown", hand_landmarks) 

            #Predict the Label
            X = getXFeatures(hand_landmarker_result.hand_landmarks[0])
            y_pred = predictLabels(sign_model, X)


        # Show the live webcam
        cv2.imshow('Rock-Paper-Scissors Recognition', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif cv2.getWindowProperty('Rock-Paper-Scissors Recognition', cv2.WND_PROP_VISIBLE) < 1:
            break
        elif key in [ord('r'), ord('p'), ord('s')]:
            if hand_landmarker_result and hand_landmarker_result.hand_landmarks:
                label = {ord('r'): "rock", ord('p'): "paper", ord('s'): "scissors"}[key]
                save_landmarks_to_csv(label, hand_landmarker_result.hand_landmarks[0])
                print(f"[SAVED] {label.upper()} sample recorded.")
            else:
                print("[WARNING] No hand detected. Try again.")        

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    draw_landmark = mp.solutions.drawing_utils  # drawing tools
    draw_styles = mp.solutions.drawing_styles # drawing colours
    mp_hands = mp.solutions.hands
    detector = None
    hand_landmarker_result = None


    # Load the trained model
    sign_model =  tf.keras.models.load_model('model/sign_model.keras')
    main()