import os
import csv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time

detector = None
hand_landmarker_result = None

# Draw landmarks on the frame
def draw_landmarks(frame, landmarks):
    for lm in landmarks:
        h, w, _ = frame.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

def print_result(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    #hand_landmarker_result = result
    global hand_landmarker_result
    hand_landmarker_result = result
    #print("Hand Landmarker Result: {}".format(result))

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
        #print(f"✅ Saved {label} sample")

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
        5. TO-DO 1: Visualize the result
        """
        # Draw landmarks if detected
        if hand_landmarker_result and hand_landmarker_result.hand_landmarks:
            draw_landmarks(frame, hand_landmarker_result.hand_landmarks[0])


        # Show the live webcam
        cv2.imshow('Rock-Paper-Scissors Recognition', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in [ord('r'), ord('p'), ord('s')]:
            if hand_landmarker_result and hand_landmarker_result.hand_landmarks:
                label = {ord('r'): "rock", ord('p'): "paper", ord('s'): "scissors"}[key]
                save_landmarks_to_csv(label, hand_landmarker_result.hand_landmarks[0])
                print(f"[SAVED] {label.upper()} sample recorded.")
            else:
                print("[WARNING] No hand detected. Try again.")

        '''# Close the window
        if cv2.waitKey(1) == ord('q'):
            break
        if cv2.getWindowProperty('Rock-Paper-Scissors Recognition', cv2.WND_PROP_VISIBLE) < 1:
            break'''

    cam.release()
    cv2.destroyAllWindows()

    """
    TO-DO 2: Save hand landmarks as dataset
    """

if __name__ == "__main__":
    main()