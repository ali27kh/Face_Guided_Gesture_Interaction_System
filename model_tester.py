import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
import pyautogui
import time
import tkinter as tk
import logging
import os
import sys

# Suppress MediaPipe warnings
logging.getLogger('mediapipe').setLevel(logging.ERROR)
sys.stderr = open(os.devnull, 'w')

class ModelTester:
    def __init__(self, root):
        self.root = root
        self.running = False
        self.cap = None
        self.model = None
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.prev_x = None
        self.prev_y = None
        self.sensitivity = 10.0
        self.alpha = 0.2
        self.dead_zone = 0.01
        self.scale_factor = 1.5
        self.last_key_press_time = 0
        self.key_press_cooldown = 0.5
        self.window_name = 'Head Mouse Control and Body Language Detection'
        self.after_id = None  # Track the after ID

    def start(self):
        if self.running:
            return
        self.running = True

        # Set up pyautogui
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0
        self.screen_width, self.screen_height = pyautogui.size()

        # Load the trained model
        try:
            with open('body_language.pkl', 'rb') as f:
                self.model = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("Model file 'body_language.pkl' not found. Please train the model first.")

        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open webcam.")

        # Get initial frame dimensions
        ret, frame = self.cap.read()
        if not ret:
            self.cleanup()
            raise RuntimeError("Failed to read initial frame from webcam.")
        self.cam_height, self.cam_width = frame.shape[:2]

        # Initialize holistic model
        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # Initialize mouse position
        ret, frame = self.cap.read()
        if ret:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = self.holistic.process(image)
            if results.face_landmarks:
                nose_tip = results.face_landmarks.landmark[1]
                init_x = nose_tip.x * self.screen_width
                init_y = nose_tip.y * self.screen_height
                pyautogui.moveTo(init_x, init_y)
                self.prev_x = nose_tip.x * self.cam_width
                self.prev_y = nose_tip.y * self.cam_height

        # Create OpenCV window explicitly
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        # Start the processing loop
        self.process_frame()

    def process_frame(self):
        if not self.running:
            self.cleanup()
            return

        ret, frame = self.cap.read()
        if not ret:
            self.cleanup()
            return

        # Process frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks
        self.mp_drawing.draw_landmarks(
            image, results.face_landmarks, self.mp_face_mesh.FACEMESH_TESSELATION,
            self.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            self.mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
        )

        try:
            if results.face_landmarks:
                face = results.face_landmarks.landmark
                face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility]
                                         for landmark in face]).flatten())
                X = pd.DataFrame([face_row])
                body_language_class = self.model.predict(X)[0]
                body_language_prob = self.model.predict_proba(X)[0]

                # Keyboard control
                current_time = time.time()
                if current_time - self.last_key_press_time > self.key_press_cooldown:
                    if body_language_class == 'class enter':
                        pyautogui.press('enter')
                        self.last_key_press_time = current_time
                    elif body_language_class == 'class space':
                        pyautogui.press('space')
                        self.last_key_press_time = current_time
                    elif body_language_class == 'class click':
                        pyautogui.click()
                        self.last_key_press_time = current_time

                # Display class and probability
                if body_language_class != 'class others':
                    coords = tuple(np.multiply(
                        np.array((results.face_landmarks.landmark[1].x, results.face_landmarks.landmark[1].y)),
                        [self.cam_width, self.cam_height]).astype(int))
                    cv2.rectangle(image, (coords[0], coords[1] + 5),
                                  (coords[0] + len(str(body_language_class)) * 20, coords[1] - 30),
                                  (245, 117, 16), -1)
                    cv2.putText(image, str(body_language_class), coords,
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)
                    cv2.putText(image, 'CLASS', (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(body_language_class).split(' ')[0],
                                (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, 'PROB', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)], 2)),
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Mouse control
                nose_tip = results.face_landmarks.landmark[1]
                center_x = nose_tip.x * self.cam_width
                center_y = nose_tip.y * self.cam_height

                if self.prev_x is not None and self.prev_y is not None:
                    dx = (center_x - self.prev_x) * self.sensitivity
                    dy = (center_y - self.prev_y) * self.sensitivity
                    if abs(dx) < self.dead_zone * self.cam_width:
                        dx = 0
                    if abs(dy) < self.dead_zone * self.cam_height:
                        dy = 0
                    smoothed_dx = self.alpha * dx + (1 - self.alpha) * (self.prev_x - center_x)
                    smoothed_dy = self.alpha * dy + (1 - self.alpha) * (self.prev_y - center_y)
                    current_mouse_x, current_mouse_y = pyautogui.position()
                    new_x = current_mouse_x + smoothed_dx
                    new_y = current_mouse_y + smoothed_dy
                    new_x = max(0, min(new_x, self.screen_width - 1))
                    new_y = max(0, min(new_y, self.screen_height - 1))
                    pyautogui.moveTo(new_x, new_y)
                self.prev_x, self.prev_y = center_x, center_y
                cv2.circle(image, (int(center_x), int(center_y)), 3, (0, 0, 255), -1)

        except Exception as e:
            print(f"Error in processing frame: {e}")

        image = cv2.resize(image, (int(self.cam_width * self.scale_factor), int(self.cam_height * self.scale_factor)))
        cv2.imshow(self.window_name, image)

        # Check for 'q' keypress with focus on the OpenCV window
        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 1:
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                self.running = False
                self.cleanup()
                return

        # Schedule the next frame only if running
        if self.running:
            self.after_id = self.root.after(10, self.process_frame)

    def cleanup(self):
        try:
            if self.cap and self.cap.isOpened():
                self.cap.release()
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow(self.window_name)
        except Exception as e:
            print(f"Error during cleanup: {e}")
        finally:
            self.running = False
            self.holistic = None
            self.model = None
            self.cap = None
            # Cancel the scheduled after call if it exists
            if self.after_id:
                self.root.after_cancel(self.after_id)
                self.after_id = None