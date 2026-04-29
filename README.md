# Adaptive-Headlights-Control-With-Machine-Learning

This project implements an intelligent, real-time system for automatic headlight control (High Beam/Low Beam), trained on the ML Car Headlight dataset from Roboflow, using Machine Learning and Computer Vision. The system is designed to enhance night-time driving safety by automatically switching to Low Beam when a vehicle is detected and returning to High Beam when the road is clear.

Key Features:

• Object Detection: Uses a trained YOLOv8 model to identify cars and light sources (headlights/taillights) in real-time. 
• Decision Logic: Automatically manages headlight states based on the presence or absence of traffic. 
• Flicker Prevention: Includes a hysteresis-based delay mechanism to stabilize the light state during brief detection gaps or high-speed movement. 
• Real-time Processing: Optimized using OpenCV and PyTorch to process video frames with minimal latency.
