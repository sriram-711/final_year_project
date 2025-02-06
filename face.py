import face_recognition
import cv2
import numpy as np

# Step 1: Load known face images and encode them
# Load an image of the person and get their face encoding
image_of_person = face_recognition.load_image_file("/Users/sriram/Desktop/person.jpeg")
person_face_encoding = face_recognition.face_encodings(image_of_person)[0]

# Step 2: Store encodings and names in lists
known_face_encodings = [person_face_encoding]
known_face_names = ["sriram"]  # Replace with the actual person's name

# Step 3: Start video capture
video_capture = cv2.VideoCapture(0)  # 0 is the default webcam

while True:
    # Step 4: Capture each frame from the webcam
    ret, frame = video_capture.read()

    # Step 5: Find all faces and their encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Step 6: Loop through each face found in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Step 7: Compare the detected face with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"  # Default name is "Unknown" if no match is found

        # If a match is found, use the corresponding name
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Step 8: Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Step 9: Add the name label below the face
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Step 10: Display the frame with the detected faces and names
    cv2.imshow('Video', frame)

    # Step 11: Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Step 12: Release the webcam and close the OpenCV window
video_capture.release()
cv2.destroyAllWindows()

