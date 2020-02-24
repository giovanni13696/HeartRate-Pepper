# HeartRate-Pepper
Heart rate measurement method using a NAO Robot and Face Detection

This project has been developed to detect the frequency of heart rate using face detection. In particulare, face detection has been made using a camera of a NAO Robot called Pepper. 

# Method

1) Capture video using Pepper's frontal camera
2) Detect face, align and get ROI using facial landmarks
3) Extract color from ROIs
4) Collect average color values in a data buffer
5) Calculate FFT of data buffer, the highest peak is heart rate 
6) Calculate average value of an array where bpms of each frame are collected


# Requirements

To get this project working you need to install all the needed libraries:
numpy, opencv, dlib, imutils, os, qi (for Peppers'API).
You also need to download "shape_predictor_68_face_landmarks.dat".


# Implementation 

Install needed libraries and run main() method in main.py
Then, place your face in front of Pepper's camera and wait for about 1 minute. Pepper will say you heart rate automatically.
