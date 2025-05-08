# GodVIewAI
### README: Patient Monitoring System

---

## **Overview**
This project implements a **real-time patient monitoring system** using **YOLOv8** for object detection and **cvzone** for visualization. The system detects individuals, classifies their posture (e.g., standing, sitting, or fallen), and displays the results on a video feed.

---

## **Features**
- **Real-Time Detection**: Detects objects in the video feed using YOLOv8.
- **Posture Classification**:
  - **Standing**: Aspect ratio > 1.5.
  - **Sitting**: 1.0 < Aspect ratio ≤ 1.5.
  - **Fall Detected**: Aspect ratio < 0.7 and position exceeds a threshold.
- **Bounding Boxes**: Draws bounding boxes around detected individuals.
- **Fall Alerts**: Displays a warning when a fall is detected.

---

## **Requirements**
### **Python Version**
- Python 3.8 or higher

---

## **Setup**
1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd GodViewAI
   ```

2. **Dependencies**
   Install the required Python libraries listed in requirements.txt:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Program**
   Execute the main script:
   ```bash
   python main.py
   ```

---