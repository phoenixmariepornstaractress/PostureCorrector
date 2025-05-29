import cv2
import mediapipe as mp
import numpy as np
import time
import os
import csv
import configparser # New import for configuration management
from collections import deque
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Try to import playsound, if not available, provide a dummy function
try:
    from playsound import playsound
except ImportError:
    print("playsound library not found. Sound alerts will be disabled.")
    def playsound(sound_file):
        print(f"Attempted to play sound: {sound_file} (playsound not installed)")


class PostureCorrector:
    def __init__(self, camera_index=0):
        # --- Configuration Management ---
        self.config_file = "config.ini"
        self.config = configparser.ConfigParser()
        self._load_config() # Load configuration settings

        # Use settings from config or defaults
        self.alert_cooldown = self.config.getint('Settings', 'alert_cooldown', fallback=10)
        self.recalibration_interval = self.config.getint('Settings', 'recalibration_interval', fallback=300)
        self.sound_file = self.config.get('Settings', 'alert_sound_file', fallback="alert.wav")
        # Camera index will be passed directly or default to 0 if not in config
        self.camera_index = camera_index # Use the provided camera_index

        # Configuration for angle history and plotting
        self.angle_window = 100  # Number of recent frames to display in the plot
        self.shoulder_vals = deque(maxlen=self.angle_window) # Stores recent shoulder angles
        self.neck_vals = deque(maxlen=self.angle_window)     # Stores recent neck angles
        self.face_vals = deque(maxlen=self.angle_window)     # Stores recent face angles

        # Setup for real-time plot using matplotlib
        self.fig, self.ax = plt.subplots(figsize=(8, 4)) # Create a figure and an axes for the plot
        self.lines = {
            "shoulder": self.ax.plot([], [], label="Shoulder Angle")[0], # Line for shoulder angles
            "neck": self.ax.plot([], [], label="Neck Angle")[0],         # Line for neck angles
            "face": self.ax.plot([], [], label="Face Angle")[0],          # Line for face angles
            "shoulder_threshold": self.ax.axhline(y=0, color='r', linestyle='--', label="Shoulder Threshold"), # New: Shoulder threshold line
            "neck_threshold": self.ax.axhline(y=0, color='g', linestyle='--', label="Neck Threshold")         # New: Neck threshold line
        }
        self.ax.set_ylim(0, 180) # Y-axis limits for angles (0 to 180 degrees)
        self.ax.set_xlim(0, self.angle_window) # X-axis limits for frames
        self.ax.set_title("Real-time Posture Angles") # Plot title
        self.ax.set_xlabel("Frame") # X-axis label
        self.ax.set_ylabel("Angle (degrees)") # Y-axis label
        self.ax.legend() # Display legend for the lines
        plt.ion() # Turn on interactive mode for matplotlib
        plt.show(block=False) # Show the plot without blocking the main execution

        # MediaPipe Pose setup
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(static_image_mode=False,
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)

        # OpenCV video capture setup
        self.cap = cv2.VideoCapture(self.camera_index) # Use camera_index from config/argument
        if not self.cap.isOpened():
            raise RuntimeError(f"Camera with index {self.camera_index} not accessible.")

        # Calibration variables
        self.calibration_frames = 0 # Counter for calibration frames
        self.shoulder_angles_cal = []   # List to store shoulder angles during calibration
        self.neck_angles_cal = []       # List to store neck angles during calibration
        self.shoulder_threshold = 0 # Dynamic threshold for shoulder angle
        self.neck_threshold = 0     # Dynamic threshold for neck angle
        self.is_calibrated = False  # Flag to indicate if calibration is complete

        # Posture monitoring and alert variables
        self.angle_history = deque(maxlen=100) # History of angles for adaptive thresholding
        self.last_alert_time = 0               # Timestamp of the last alert

        # Logging and training data files
        self.log_file = "posture_log.csv"
        self.training_data_file = "training_data.csv"
        self._initialize_log_file() # Initialize the log file if it doesn't exist

        # K-Nearest Neighbors (KNN) classifier setup
        self.knn = KNeighborsClassifier(n_neighbors=3) # KNN classifier with 3 neighbors
        self.knn_trained = False                       # Flag to indicate if KNN is trained
        self.X_train = []                              # Training features (angles)
        self.y_train = []                              # Training labels (posture status)
        self._load_training_data()                     # Load existing training data

        # Recalibration settings
        self.last_recalibration_time = time.time() # Timestamp of the last recalibration

        # Session statistics
        self.session_start_time = time.time()
        self.posture_status_counts = {"Good Posture": 0, "Poor Posture": 0, "Unknown": 0}
        self.frame_count = 0

        # Pause functionality
        self.paused = False
        self.pause_start_time = 0

    def _load_config(self):
        """Loads configuration from config.ini file."""
        self.config.read(self.config_file)
        if not self.config.has_section('Settings'):
            self.config.add_section('Settings')
            # Set default values if section doesn't exist
            self.config.set('Settings', 'alert_cooldown', '10')
            self.config.set('Settings', 'recalibration_interval', '300')
            self.config.set('Settings', 'alert_sound_file', 'alert.wav')
            # camera_index is handled by the constructor argument, not stored in config by default
            self._save_config() # Save default config if not found

    def _save_config(self):
        """Saves current configuration to config.ini file."""
        with open(self.config_file, 'w') as configfile:
            self.config.write(configfile)
        print(f"Configuration saved to {self.config_file}")

    def _initialize_log_file(self):
        """Initializes the CSV log file with headers if it does not already exist."""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as file:
                csv.writer(file).writerow(["Timestamp", "Shoulder Angle", "Neck Angle", "Face Angle", "Status"])

    def _calculate_angle(self, a, b, c):
        """
        Calculates the angle (in degrees) between three points.
        Args:
            a, b, c (tuple): Coordinates of the three points (x, y).
        Returns:
            float: The calculated angle in degrees.
        """
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        # Clip cosine_angle to avoid numerical errors that might result in values slightly outside [-1, 1]
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def _draw_angle(self, image, a, b, c, angle, color):
        """
        Draws lines connecting three points and displays the calculated angle on the image.
        Args:
            image (numpy.array): The OpenCV image frame.
            a, b, c (tuple): Coordinates of the three points (x, y).
            angle (float): The angle to display.
            color (tuple): BGR color for drawing.
        """
        cv2.line(image, a, b, color, 2) # Draw line from a to b
        cv2.line(image, b, c, color, 2) # Draw line from b to c
        cv2.putText(image, f"{int(angle)} deg", (b[0] + 10, b[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2) # Display angle text

    def _log_posture(self, timestamp, shoulder_angle, neck_angle, face_angle, status):
        """
        Logs the current posture data to the CSV log file.
        Args:
            timestamp (str): Current timestamp.
            shoulder_angle (float): Calculated shoulder angle.
            neck_angle (float): Calculated neck angle.
            face_angle (float): Calculated face angle.
            status (str): Posture status (e.g., "Good Posture", "Poor Posture").
        """
        with open(self.log_file, 'a', newline='') as file:
            csv.writer(file).writerow([timestamp, shoulder_angle, neck_angle, face_angle, status])

    def _reset_calibration(self):
        """Resets all calibration-related variables."""
        self.calibration_frames = 0
        self.shoulder_angles_cal.clear()
        self.neck_angles_cal.clear()
        self.shoulder_threshold = 0
        self.neck_threshold = 0
        self.is_calibrated = False
        # Reset threshold lines on plot
        self.lines["shoulder_threshold"].set_ydata([0])
        self.lines["neck_threshold"].set_ydata([0])
        print("Calibration reset.")

    def _adaptive_thresholds(self):
        """
        Calculates adaptive thresholds for shoulder and neck angles based on recent history.
        This helps in dynamically adjusting to the user's natural posture.
        """
        if len(self.angle_history) >= 30: # Ensure enough data for meaningful statistics
            history_array = np.array(self.angle_history)
            means = np.mean(history_array, axis=0) # Mean of shoulder and neck angles
            stds = np.std(history_array, axis=0)   # Standard deviation of shoulder and neck angles
            # Thresholds are set as mean minus one standard deviation.
            # A lower angle than this threshold indicates a deviation.
            self.shoulder_threshold = means[0] - stds[0]
            self.neck_threshold = means[1] - stds[1]
            # Update threshold lines on plot
            self.lines["shoulder_threshold"].set_ydata([self.shoulder_threshold])
            self.lines["neck_threshold"].set_ydata([self.neck_threshold])


    def _train_classifier(self):
        """Trains the KNN classifier if sufficient training data is available."""
        if len(self.X_train) >= 10: # Require at least 10 samples to train
            self.knn.fit(self.X_train, self.y_train)
            self.knn_trained = True
            print("KNN classifier trained.")
        else:
            print(f"Need at least 10 training samples to train KNN. Currently have {len(self.X_train)}.")

    def _save_training_sample(self, shoulder, neck, face, label):
        """
        Saves a single training sample (angles and label) to the training data CSV file.
        Args:
            shoulder (float): Shoulder angle.
            neck (float): Neck angle.
            face (float): Face angle.
            label (str): Posture label (e.g., "Good", "Poor").
        """
        with open(self.training_data_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([shoulder, neck, face, label])
        print(f"Saved training sample: Shoulder={shoulder:.1f}, Neck={neck:.1f}, Face={face:.1f}, Label={label}")

    def _load_training_data(self):
        """Loads existing training data from the CSV file and trains the KNN classifier."""
        if os.path.exists(self.training_data_file):
            with open(self.training_data_file, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    if len(row) == 4: # Ensure row has expected number of columns
                        try:
                            shoulder, neck, face = map(float, row[:3])
                            label = row[3]
                            self.X_train.append([shoulder, neck, face])
                            self.y_train.append(label)
                        except ValueError as e:
                            print(f"Skipping malformed row in training data: {row} - {e}")
            if len(self.X_train) >= 10: # Attempt to train after loading
                self._train_classifier()
            else:
                print(f"Loaded {len(self.X_train)} training samples. Not enough to train KNN yet.")

    def _play_alert(self):
        """Plays the alert sound if the file exists."""
        if os.path.exists(self.sound_file):
            try:
                playsound(self.sound_file)
            except Exception as e:
                print(f"Failed to play sound: {e}")
        else:
            print(f"Alert sound file not found at: {self.sound_file}")

    def _update_plot(self):
        """Updates the real-time matplotlib plot with the latest angle data."""
        # Update data for each line
        self.lines["shoulder"].set_data(range(len(self.shoulder_vals)), list(self.shoulder_vals))
        self.lines["neck"].set_data(range(len(self.neck_vals)), list(self.neck_vals))
        self.lines["face"].set_data(range(len(self.face_vals)), list(self.face_vals))

        # Adjust X-axis limits to show only the last 'angle_window' frames
        self.ax.set_xlim(max(0, len(self.shoulder_vals) - self.angle_window), len(self.shoulder_vals))

        # Redraw the canvas
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _display_session_summary(self):
        """Calculates and prints a summary of the posture session."""
        session_duration_sec = time.time() - self.session_start_time
        if self.paused: # Adjust duration if session ended while paused
            session_duration_sec -= (time.time() - self.pause_start_time)
        session_duration_min = session_duration_sec / 60

        print("\n--- Session Summary ---")
        print(f"Session Duration: {session_duration_min:.2f} minutes")
        print(f"Total Frames Processed: {self.frame_count}")

        total_classified_frames = sum(self.posture_status_counts.values())
        if total_classified_frames > 0:
            for status, count in self.posture_status_counts.items():
                percentage = (count / total_classified_frames) * 100
                print(f"- {status}: {count} frames ({percentage:.2f}%)")
        else:
            print("No posture data was classified during this session.")
        print("-----------------------")


    def run(self):
        """Main loop for capturing video, processing posture, and displaying results."""
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                # If frame not captured, wait a bit and continue to next iteration
                time.sleep(0.01)
                continue

            # Flip frame horizontally for a more intuitive mirror view
            frame = cv2.flip(frame, 1)

            # --- Pause/Resume Logic ---
            if self.paused:
                cv2.putText(frame, "PAUSED", (frame.shape[1] // 2 - 50, frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow("Posture Corrector", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('p'):
                    self.paused = False
                    # Adjust session start time to account for pause duration
                    self.session_start_time += (time.time() - self.pause_start_time)
                    print("Resumed posture detection.")
                elif key == ord('q'):
                    break
                continue # Skip processing frame if paused

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert to RGB for MediaPipe
            result = self.pose.process(rgb_frame) # Process the frame with MediaPipe Pose

            self.frame_count += 1 # Increment frame count for session summary

            if result.pose_landmarks:
                lm = result.pose_landmarks.landmark # Get detected landmarks
                h, w = frame.shape[:2] # Get frame height and width

                # Extract coordinates for key body parts
                left_shoulder = int(lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w), \
                                int(lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h)
                right_shoulder = int(lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w), \
                                 int(lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h)
                left_ear = int(lm[self.mp_pose.PoseLandmark.LEFT_EAR.value].x * w), \
                           int(lm[self.mp_pose.PoseLandmark.LEFT_EAR.value].y * h)
                nose = int(lm[self.mp_pose.PoseLandmark.NOSE.value].x * w), \
                       int(lm[self.mp_pose.PoseLandmark.NOSE.value].y * h)

                # Calculate angles
                # Shoulder angle: Angle between left shoulder, right shoulder, and a point vertically below right shoulder
                shoulder_angle = self._calculate_angle(left_shoulder, right_shoulder, (right_shoulder[0], right_shoulder[1] + 100))
                # Neck angle: Angle between left ear, left shoulder, and a point vertically below left shoulder
                neck_angle = self._calculate_angle(left_ear, left_shoulder, (left_shoulder[0], left_shoulder[1] + 100))
                # Face angle: Angle between nose, left ear, and a point vertically below left ear
                face_angle = self._calculate_angle(nose, left_ear, (left_ear[0], left_ear[1] + 100))


                # Append current angles to history for plotting
                self.shoulder_vals.append(shoulder_angle)
                self.neck_vals.append(neck_angle)
                self.face_vals.append(face_angle)
                self._update_plot() # Update the real-time plot

                # Add current angles to history for adaptive thresholding
                self.angle_history.append((shoulder_angle, neck_angle))
                self._adaptive_thresholds() # Recalculate adaptive thresholds

                current_time = time.time()

                # Calibration phase
                if not self.is_calibrated and self.calibration_frames < 30:
                    self.shoulder_angles_cal.append(shoulder_angle)
                    self.neck_angles_cal.append(neck_angle)
                    self.calibration_frames += 1
                    cv2.putText(frame, f"Calibrating... {self.calibration_frames}/30", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                elif not self.is_calibrated:
                    self.is_calibrated = True
                    print("Calibration complete with adaptive thresholding.")

                # Automatic recalibration
                if current_time - self.last_recalibration_time > self.recalibration_interval:
                    self._reset_calibration()
                    self.last_recalibration_time = current_time
                    print("Initiating auto-recalibration...")

                # Draw angles and landmarks on the frame
                self._draw_angle(frame, left_shoulder, right_shoulder, (right_shoulder[0], right_shoulder[1] + 100), shoulder_angle, (255, 0, 0))
                self._draw_angle(frame, left_ear, left_shoulder, (left_shoulder[0], left_shoulder[1] + 100), neck_angle, (0, 255, 0))
                self._draw_angle(frame, nose, left_ear, (left_ear[0], left_ear[1] + 100), face_angle, (0, 255, 255))

                self.mp_drawing.draw_landmarks(frame, result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                status = "Unknown"
                color = (255, 255, 255) # Default text color

                # Posture classification and alerts
                if self.knn_trained:
                    # Predict posture status using the trained KNN model
                    status = self.knn.predict([[shoulder_angle, neck_angle, face_angle]])[0]
                    if status != "Good" and current_time - self.last_alert_time > self.alert_cooldown:
                        print("Alert: " + status)
                        self._play_alert()
                        self.last_alert_time = current_time
                        color = (0, 0, 255) if status == "Poor" else (0, 165, 255) # Red for Poor, Orange for other non-Good
                    elif status == "Good":
                        color = (0, 255, 0) # Green for Good posture
                else:
                    # Fallback to threshold-based detection if KNN is not trained
                    if shoulder_angle < self.shoulder_threshold or neck_angle < self.neck_threshold:
                        status = "Poor Posture"
                        color = (0, 0, 255) # Red color for poor posture
                        if current_time - self.last_alert_time > self.alert_cooldown:
                            print("Poor posture detected! Please sit up straight.")
                            self._play_alert()
                            self.last_alert_time = current_time
                    else:
                        status = "Good Posture"
                        color = (0, 255, 0) # Green color for good posture

                # Update session statistics
                self.posture_status_counts[status] = self.posture_status_counts.get(status, 0) + 1

                # Log current posture data
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                self._log_posture(timestamp, shoulder_angle, neck_angle, face_angle, status)

                # Display status and angle values on the frame
                cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f"Shoulder: {shoulder_angle:.1f}/{self.shoulder_threshold:.1f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"Neck: {neck_angle:.1f}/{self.neck_threshold:.1f}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"Face: {face_angle:.1f}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, "Press 'p' to pause", (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

            cv2.imshow("Posture Corrector", frame) # Display the processed frame

            # Handle keyboard inputs
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): # Quit the application
                break
            elif key == ord('r'): # Reset calibration
                self._reset_calibration()
            elif key == ord('t'): # Train classifier with current posture
                if result.pose_landmarks:
                    print("Enter label for current posture (e.g., Good, Poor, Leaning): ", end="")
                    label = input().strip()
                    if label: # Only save if a label is provided
                        self.X_train.append([shoulder_angle, neck_angle, face_angle])
                        self.y_train.append(label)
                        self._train_classifier() # Retrain KNN with new data
                        self._save_training_sample(shoulder_angle, neck_angle, face_angle, label)
                    else:
                        print("Label cannot be empty. Training sample not saved.")
                else:
                    print("No pose detected. Cannot train without valid posture angles.")
            elif key == ord('p'): # Pause/Resume
                self.paused = True
                self.pause_start_time = time.time()
                print("Paused posture detection. Press 'p' again to resume.")


        # Release camera and destroy all OpenCV windows on exit
        self.cap.release()
        cv2.destroyAllWindows()
        plt.close(self.fig) # Close the matplotlib plot window
        self._display_session_summary() # Display session summary before exiting


if __name__ == "__main__":
    print("Enter camera index (default is 0): ", end="")
    try:
        index = int(input())
    except ValueError:
        index = 0 # Default to 0 if input is not an integer
        print("Invalid input. Defaulting to camera index 0.")
    except Exception as e:
        index = 0
        print(f"An unexpected error occurred during input: {e}. Defaulting to camera index 0.")

    try:
        corrector = PostureCorrector(camera_index=index)
        corrector.run()
    except RuntimeError as e:
        print(f"Error: {e}. Please ensure your camera is connected and accessible.")
    except Exception as e:
        print(f"An unexpected error occurred during execution: {e}")


