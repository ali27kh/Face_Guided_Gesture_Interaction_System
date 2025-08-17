import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import csv
import os
import numpy as np


# Initialize MediaPipe drawing and holistic model
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh

def collect_landmarks(class_name, csv_file='coords.csv'):
    """
    Opens the camera, captures face landmarks, and saves them to a CSV file with the specified class name.
    
    Args:
        class_name (str): The class label to prepend to each row in the CSV.
        csv_file (str): Path to the CSV file where landmarks will be saved. Default is 'coords.csv'.
    """
    # Initialize CSV file with header (only if file doesn't exist)
    try:
        with open(csv_file, mode='x', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            # Create header for face landmarks (468 landmarks * 4 values: x, y, z, visibility)
            header = ['class'] + [f'face_{i}_{attr}' for i in range(468) for attr in ['x', 'y', 'z', 'visibility']]
            csv_writer.writerow(header)
    except FileExistsError:
        pass  # File already exists, skip header creation

    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Initialize holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                break

            # Recolor feed to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detections
            results = holistic.process(image)

            # Recolor image back to BGR for rendering
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw landmarks
            # 1. Face landmarks
            mp_drawing.draw_landmarks(
                image,
                results.face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
            )

            # 2. Right hand
            mp_drawing.draw_landmarks(
                image,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
            )

            # 3. Left hand
            mp_drawing.draw_landmarks(
                image,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
            )

            # 4. Pose detections
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

            # Export face landmarks to CSV
            try:
                if results.face_landmarks:
                    # Extract face landmarks
                    face = results.face_landmarks.landmark
                    face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] 
                                             for landmark in face]).flatten())
                    
                    # Prepend class name
                    row = [class_name] + face_row

                    # Append to CSV
                    with open(csv_file, mode='a', newline='') as f:
                        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow(row)
            except Exception as e:
                print(f"Error writing to CSV: {e}")

            # Display the image
            cv2.imshow('Raw Webcam Feed', image)

            # Break loop on 'q' key press
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()