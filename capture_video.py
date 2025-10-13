import cv2

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
    
    # Show the live webcam
    cv2.imshow('Rock-Paper-Scissors Recognition', frame)

    # Close the window
    if cv2.waitKey(1) == ord('q'):
        break

    if cv2.getWindowProperty('Rock-Paper-Scissors Recognition', cv2.WND_PROP_VISIBLE) < 1:
        break

cam.release()
cv2.destroyAllWindows()