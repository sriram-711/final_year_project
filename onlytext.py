import cv2
import pytesseract
import pyttsx3
import time

# Configure pytesseract executable path (required on Windows)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Uncomment and adjust the path if needed

# Initialize pyttsx3 TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust speed
engine.setProperty('volume', 1)  # Adjust volume (0.0 to 1.0)

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

while True:
    # Capture frame-by-frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Convert the frame to grayscale (helps with text detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use Tesseract to detect text from the frame
    detected_text = pytesseract.image_to_string(gray)

    # If some text is detected, speak it out loud
    if detected_text.strip():  # Only speak if there's any detected text
        print(f"Detected Text: {detected_text.strip()}")
        engine.say(detected_text)  # Convert the text to speech
        engine.runAndWait()  # Wait until speech finishes

    # Display the frame with text detection (for debugging)
    cv2.putText(frame, "Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow("Live Text Detection", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
