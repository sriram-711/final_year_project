import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import pyttsx3  # Import the pyttsx3 library for text-to-speech

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Adjust the speed of speech
tts_engine.setProperty('volume', 1)  # Set volume to max

MARGIN = -1  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
rect_color = (255, 0, 255)
TEXT_COLOR = (255, 0, 0)  # red
cap = cv2.VideoCapture(0)

# Load the object detection model
base_options = python.BaseOptions(model_asset_path='indiamoney.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

# Keep track of the last detected object
last_detected = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read from the camera.")
        break

    # Flip the frame horizontally for natural viewing
    frame = cv2.flip(frame, 1)

    # Convert frame to MediaPipe image format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # Perform object detection
    detection_result = detector.detect(mp_image)

    for detection in detection_result.detections:
        # Get bounding box
        bbox = detection.bounding_box
        x = int(bbox.origin_x)
        y = int(bbox.origin_y)
        w = int(bbox.width)
        h = int(bbox.height)

        # Draw bounding box
        start_point = (x, y)
        end_point = (x + w, y + h)
        cv2.rectangle(frame, start_point, end_point, rect_color, 3)

        # Get category
        category = detection.categories[0]
        category_name = category.category_name

        # Set text location above the rectangle
        text_location = (x, y - 10)  # Adjust if necessary

        # Put text on the image
        result_text = f"{category_name}"
        cv2.putText(frame, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

        # Speak the detected object if it is a new detection
        if category_name != last_detected:
            audio_message = f"{category_name} detected."
            tts_engine.say(audio_message)  # Speak the detected object
            tts_engine.runAndWait()  # Wait until the speech is finished
            last_detected = category_name  # Update the last detected object

    # Show the frame with detections
    cv2.imshow("test window", frame)

    # Exit on pressing 'ESC'
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
