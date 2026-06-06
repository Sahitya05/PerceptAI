# 👁️ PerceptAI

[![Python](https://shields.io)](https://python.org)
[![PyQt](https://shields.io)](https://qt.io)
[![YOLOv5](https://shields.io)](https://github.com)

PerceptAI is an advanced computer vision application powered by Python and the YOLOv5 object detection framework. Featuring a modern, styled graphical user interface, this project provides seamless real-time visual perception capabilities.

---

## ✨ Features

- **Real-Time Detection:** Powered by optimized `yolov5s.pt` and `yolov5m.pt` models for varying accuracy and speed requirements.
- **Modern GUI:** Built with Python (using `sightai.py`) and customized using a beautiful, responsive stylesheet (`theme.qss`).
- **Flexible Inputs:** Designed to handle static images (`images 1.jpeg`) as well as live video processing.

---

## 📂 Project Structure

```text
Perceptai/
├── yolov5/                  # YOLOv5 core engine and dependencies
├── __pycache__/             # Compiled python binaries
├── sightai.py               # Main application entry point
├── theme.qss                # Custom stylesheet for the UI interface
├── yolov5s.pt               # YOLOv5 Small weight file
├── yolov5m.pt               # YOLOv5 Medium weight file
├── images 1.jpeg            # Sample image for testing detection
└── README.md                # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.10 or higher installed on your machine.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd PerceptAI
   ```

2. **Install core requirements:**
   *(Navigate to the yolov5 directory to install object detection dependencies)*
   ```bash
   pip install -r yolov5/requirements.txt
   ```

3. **Install UI requirements:**
   ```bash
   pip install PyQt5  # Or the specific Qt binding used in sightai.py
   ```

### Running the Application

Launch the main application interface by running:
```bash
python sightai.py
```

---

## 🛠️ Built With

- **[Python](https://python.org)** - Core logic and scripting.
- **[YOLOv5 by Ultralytics](https://github.com)** - State-of-the-art object detection.
- **[Qt/PyQt](https://qt.io)** - High-quality frontend interface framework.


---
*Developed as part of the 6th Semester College Curriculum *
