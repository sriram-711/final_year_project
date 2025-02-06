import face_recognition
import cv2
import os

# Step 1: Load and encode multiple known faces
known_face_encodings = []
known_face_names = []

# Define the path to the directory with known face images
image_directory = "/Users/sriram/Desktop/faces"

# Loop through the files in the directory and load each face
for image_filename in os.listdir(image_directory):
    image_path = os.path.join(image_directory, image_filename)
    # Skip non-image files
    if not image_filename.endswith(('.jpg', '.png', '.jpeg')):
        continue
    
    # Load the image and encode the face
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)
    
    # If there is at least one face encoding, add it to our list
    if encoding:
        known_face_encodings.append(encoding[0])
        # Use the image filename without extension as the name (e.g., "john_doe")
        name = os.path.splitext(image_filename)[0]
        known_face_names.append(name)

# Step 2: Initialize the webcam feed
video_capture = cv2.VideoCapture(0)

while True:
    # Step 3: Capture each frame from the webcam
    ret, frame = video_capture.read()

    # Step 4: Find all face locations and face encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Step 5: Loop through each detected face
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the detected face with all known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"  # Default name if no match is found

        # If a match is found, use the corresponding name
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Step 6: Draw a rectangle around the face and display the name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Step 7: Display the frame with face detections and names
    cv2.imshow('Video', frame)

    # Step 8: Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Step 9: Release the video capture object and close OpenCV windows
video_capture.release()
cv2.destroyAllWindows()

