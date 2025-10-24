import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time

detector = None
hand_landmarker_result = None

def print_result(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    hand_landmarker_result = result
    print("Hand Landmarker Result: {}".format(result))

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
        


        # Show the live webcam
        cv2.imshow('Rock-Paper-Scissors Recognition', frame)

        # Close the window
        if cv2.waitKey(1) == ord('q'):
            break
        if cv2.getWindowProperty('Rock-Paper-Scissors Recognition', cv2.WND_PROP_VISIBLE) < 1:
            break

    cam.release()
    cv2.destroyAllWindows()

    """
    TO-DO 2: Save hand landmarks as dataset
    """

if __name__ == "__main__":
    main()