import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def print_result(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print("Hand Landmarker Result: {}".format(result))

def setup_hand_landmarker():
    """
    Initialize and return a MediaPipe Hand Landmarker detector
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
    return detector

def detect_hand_landmark():
    """
    Read the frame from webcam and run the detection
    """

    ## Load Webcam ##
    """
    TO-DO: 
    1. Read frame from OpenCV 
    2. Convert frame received from Webcam to a MediaPipe's Image object
    3. Send live image data to perform hand landmarks detection
    """

    ## Detect hand landmarks from the input real-time webcam ##
    # hand_landmarker_result = landmarker.detect_async(mp_image, frame_timestamp_ms)

def main():
    detector = setup_hand_landmarker()

if __name__ == "__main__":
    main()