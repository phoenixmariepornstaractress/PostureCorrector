# PostureCorrector
PostureCorrector is a real-time posture monitoring system that utilizes computer vision and machine learning to detect and classify user posture using a webcam. 

# PostureCorrector

PostureCorrector is a real-time posture monitoring and correction tool that uses computer vision and machine learning to help users maintain healthy posture while seated or standing. The system captures body landmarks using a webcam, calculates posture-related angles, and provides instant feedback through visual cues and audio alerts.

## Features

* Real-time posture detection using OpenCV and MediaPipe
* Angle-based analysis of shoulders, neck, and face position
* Calibration and adaptive thresholding
* Visual alerts and audio notifications for poor posture
* Machine learning classification using K-Nearest Neighbors (KNN)
* Logging of posture metrics and events to CSV
* Live data visualization with Matplotlib
* Option to collect and train custom posture data

## Requirements

* Python 3.7+
* OpenCV
* MediaPipe
* NumPy
* scikit-learn
* Matplotlib
* playsound

You can install all required packages using:

```bash
pip install -r requirements.txt
```

## Usage

Run the script and follow on-screen instructions:

```bash
python posture_corrector.py
```

* Press `q` to quit the program
* Press `r` to reset calibration
* Press `t` to label and train the current posture

Ensure your webcam is properly connected and unobstructed.

## Contribution

Contributions are welcome! Whether you want to report a bug, request a feature, or contribute code/documentation, please feel free to open an issue or submit a pull request.

Before contributing:

* Fork the repository
* Create a new branch for your changes
* Follow PEP 8 style guidelines
* Test your changes thoroughly

If you have ideas to expand the project—such as integration with mobile devices, ergonomic assessment tools, or posture improvement analytics—we'd love to hear from you.

## License

This project is licensed under the MIT License.

---

Feel free to reach out if you have questions, suggestions, or want to collaborate!
 
