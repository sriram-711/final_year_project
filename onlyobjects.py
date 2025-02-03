import cv2
from ultralytics import YOLO
import pyttsx3  # Import pyttsx3 for text-to-speech

# Load the pre-trained YOLOv8 model (yolov8n.pt)
model = YOLO("yolov8n.pt")

# Initialize webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Adjust the speed of speech
tts_engine.setProperty('volume', 1)  # Set volume to max

# Keep track of the last detected object
last_detected = None

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    # Perform inference on the captured frame
    results = model(frame)

    # Access the boxes from the results
    boxes = results[0].boxes  # This contains bounding boxes, confidence, and class information
    
    # Loop through each detection (bounding box) in the frame
    for box in boxes:
        # Extract the bounding box coordinates, confidence, and class ID
        x1, y1, x2, y2 = box.xyxy[0]  # The coordinates of the bounding box (top-left and bottom-right)
        confidence = box.conf[0].item()  # Confidence score of the detection
        class_id = int(box.cls[0].item())  # Class ID (as an integer)
        label = model.names[class_id]  # The name of the detected class (e.g., "person")
        
        # Draw the bounding box and label on the frame
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # Blue color
        cv2.putText(frame, f'{label} {confidence:.2f}', (int(x1), int(y1)-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Blue color

        # Speak the detected object if it is a new detection
        if label != last_detected:
            audio_message = f"{label}."
            tts_engine.say(audio_message)  # Speak the detected object
            tts_engine.runAndWait()  # Wait until the speech is finished
            last_detected = label  # Update the last detected object

    # Display the resulting frame with detections
    cv2.imshow("YOLOv8 Live Detection", frame)

    # Press 'q' to quit the live video feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
