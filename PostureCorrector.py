import cv2
import mediapipe as mp
import numpy as np
import time
import os
import csv
from playsound import playsound
from collections import deque
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class PostureCorrector:
    def __init__(self, camera_index=0):
        self.angle_window = 100  # Number of recent frames to display
        self.shoulder_vals = deque(maxlen=self.angle_window)
        self.neck_vals = deque(maxlen=self.angle_window)
        self.face_vals = deque(maxlen=self.angle_window)

        self.fig, self.ax = plt.subplots()
        self.lines = {
            "shoulder": self.ax.plot([], [], label="Shoulder")[0],
            "neck": self.ax.plot([], [], label="Neck")[0],
            "face": self.ax.plot([], [], label="Face")[0]
        }
        self.ax.set_ylim(0, 180)
        self.ax.set_xlim(0, self.angle_window)
        self.ax.set_title("Real-time Posture Angles")
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("Angle (degrees)")
        self.ax.legend()
        plt.ion()
        plt.show()

        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(static_image_mode=False,
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)
        self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            raise RuntimeError(f"Camera with index {camera_index} not accessible.")

        self.calibration_frames = 0
        self.shoulder_angles = []
        self.neck_angles = []
        self.shoulder_threshold = 0
        self.neck_threshold = 0
        self.is_calibrated = False

        self.angle_history = deque(maxlen=100)
        self.last_alert_time = 0
        self.alert_cooldown = 10
        self.sound_file = "alert.wav"

        self.log_file = "posture_log.csv"
        self.training_data_file = "training_data.csv"
        self._initialize_log_file()

        self.knn = KNeighborsClassifier(n_neighbors=3)
        self.knn_trained = False
        self.X_train = []
        self.y_train = []
        self._load_training_data()

        self.recalibration_interval = 300  # seconds
        self.last_recalibration_time = time.time()

    def _initialize_log_file(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as file:
                csv.writer(file).writerow(["Timestamp", "Shoulder Angle", "Neck Angle", "Face Angle", "Status"])

    def _calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def _draw_angle(self, image, a, b, c, angle, color):
        cv2.line(image, a, b, color, 2)
        cv2.line(image, b, c, color, 2)
        cv2.putText(image, f"{int(angle)} deg", (b[0] + 10, b[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def _log_posture(self, timestamp, shoulder_angle, neck_angle, face_angle, status):
        with open(self.log_file, 'a', newline='') as file:
            csv.writer(file).writerow([timestamp, shoulder_angle, neck_angle, face_angle, status])

    def _reset_calibration(self):
        self.calibration_frames = 0
        self.shoulder_angles.clear()
        self.neck_angles.clear()
        self.shoulder_threshold = 0
        self.neck_threshold = 0
        self.is_calibrated = False
        print("Calibration reset.")

    def _adaptive_thresholds(self):
        if len(self.angle_history) >= 30:
            history_array = np.array(self.angle_history)
            means = np.mean(history_array, axis=0)
            stds = np.std(history_array, axis=0)
            self.shoulder_threshold = means[0] - stds[0]
            self.neck_threshold = means[1] - stds[1]

    def _train_classifier(self):
        if len(self.X_train) >= 10:
            self.knn.fit(self.X_train, self.y_train)
            self.knn_trained = True

    def _save_training_sample(self, shoulder, neck, face, label):
        with open(self.training_data_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([shoulder, neck, face, label])

    def _load_training_data(self):
        if os.path.exists(self.training_data_file):
            with open(self.training_data_file, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    if len(row) == 4:
                        shoulder, neck, face = map(float, row[:3])
                        label = row[3]
                        self.X_train.append([shoulder, neck, face])
                        self.y_train.append(label)
            if len(self.X_train) >= 10:
                self._train_classifier()

    def _play_alert(self):
        if os.path.exists(self.sound_file):
            try:
                playsound(self.sound_file)
            except Exception as e:
                print(f"Failed to play sound: {e}")
        else:
            print("Alert sound file not found.")

    def _update_plot(self):
        self.lines["shoulder"].set_data(range(len(self.shoulder_vals)), list(self.shoulder_vals))
        self.lines["neck"].set_data(range(len(self.neck_vals)), list(self.neck_vals))
        self.lines["face"].set_data(range(len(self.face_vals)), list(self.face_vals))
        self.ax.set_xlim(max(0, len(self.shoulder_vals) - self.angle_window), len(self.shoulder_vals))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def run(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.pose.process(rgb_frame)

            if result.pose_landmarks:
                lm = result.pose_landmarks.landmark
                h, w = frame.shape[:2]

                left_shoulder = int(lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w), \
                                int(lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h)
                right_shoulder = int(lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w), \
                                 int(lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h)
                left_ear = int(lm[self.mp_pose.PoseLandmark.LEFT_EAR.value].x * w), \
                           int(lm[self.mp_pose.PoseLandmark.LEFT_EAR.value].y * h)
                nose = int(lm[self.mp_pose.PoseLandmark.NOSE.value].x * w), \
                       int(lm[self.mp_pose.PoseLandmark.NOSE.value].y * h)

                shoulder_angle = self._calculate_angle(left_shoulder, right_shoulder, (right_shoulder[0], 0))
                neck_angle = self._calculate_angle(left_ear, left_shoulder, (left_shoulder[0], 0))
                face_angle = self._calculate_angle(nose, left_ear, (left_ear[0], 0))

                self.shoulder_vals.append(shoulder_angle)
                self.neck_vals.append(neck_angle)
                self.face_vals.append(face_angle)
                self._update_plot()

                self.angle_history.append((shoulder_angle, neck_angle))
                self._adaptive_thresholds()

                current_time = time.time()
                if not self.is_calibrated and self.calibration_frames < 30:
                    self.shoulder_angles.append(shoulder_angle)
                    self.neck_angles.append(neck_angle)
                    self.calibration_frames += 1
                    cv2.putText(frame, f"Calibrating... {self.calibration_frames}/30", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                elif not self.is_calibrated:
                    self.is_calibrated = True
                    print("Calibration complete with adaptive thresholding.")

                if current_time - self.last_recalibration_time > self.recalibration_interval:
                    self._reset_calibration()
                    self.last_recalibration_time = current_time

                midpoint = ((left_shoulder[0] + right_shoulder[0]) // 2,
                            (left_shoulder[1] + right_shoulder[1]) // 2)

                self._draw_angle(frame, left_shoulder, midpoint, (midpoint[0], 0), shoulder_angle, (255, 0, 0))
                self._draw_angle(frame, left_ear, left_shoulder, (left_shoulder[0], 0), neck_angle, (0, 255, 0))
                self._draw_angle(frame, nose, left_ear, (left_ear[0], 0), face_angle, (0, 255, 255))

                self.mp_drawing.draw_landmarks(frame, result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                status = "Unknown"
                color = (255, 255, 255)

                if self.knn_trained:
                    status = self.knn.predict([[shoulder_angle, neck_angle, face_angle]])[0]
                    if status != "Good" and current_time - self.last_alert_time > self.alert_cooldown:
                        print("Alert: " + status)
                        self._play_alert()
                        self.last_alert_time = current_time
                else:
                    if shoulder_angle < self.shoulder_threshold or neck_angle < self.neck_threshold:
                        status = "Poor Posture"
                        color = (0, 0, 255)
                        if current_time - self.last_alert_time > self.alert_cooldown:
                            print("Poor posture detected! Please sit up straight.")
                            self._play_alert()
                            self.last_alert_time = current_time
                    else:
                        status = "Good Posture"
                        color = (0, 255, 0)

                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                self._log_posture(timestamp, shoulder_angle, neck_angle, face_angle, status)

                cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f"Shoulder: {shoulder_angle:.1f}/{self.shoulder_threshold:.1f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"Neck: {neck_angle:.1f}/{self.neck_threshold:.1f}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"Face: {face_angle:.1f}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("Posture Corrector", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('r'):
                self._reset_calibration()
            elif key == ord('t'):
                if result.pose_landmarks:
                    print("Enter label for current posture: ", end="")
                    label = input().strip()
                    self.X_train.append([shoulder_angle, neck_angle, face_angle])
                    self.y_train.append(label)
                    self._train_classifier()
                    self._save_training_sample(shoulder_angle, neck_angle, face_angle, label)
                else:
                    print("No pose detected. Cannot train without valid posture angles.")

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Enter camera index (default is 0): ", end="")
    try:
        index = int(input())
    except:
        index = 0
    try:
        PostureCorrector(camera_index=index).run()
    except Exception as e:
        print(f"Error starting posture corrector: {e}")
