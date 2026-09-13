# Social Distance Detection using YOLOv3

A real-time social distancing monitoring system built with YOLOv3 and SSD MobileNet human detection models, with a Streamlit interface for model selection.

## Overview

This project detects people in video streams and estimates real-world distances between them to flag proximity risk. Rather than relying on raw pixel distance (which distorts with camera angle and depth), it applies a **perspective transformation** to map the scene into a bird's-eye view, then calculates calibrated **Euclidean distances** in that transformed space to classify individuals into high, low, and no-risk proximity zones.

## Key Features

- **Dual detection models**: YOLOv3 and SSD MobileNet, selectable through a Streamlit interface
- **Bird's-eye view mapping**: perspective transformation to correct for camera angle before distance calculation
- **Real-world distance calibration**: converts pixel distances into real-world units for accurate proximity classification
- **Risk zone classification**: individuals are grouped into high, low, and no-risk categories based on calibrated distance thresholds

## Results

- **92% detection accuracy** with a pre-trained YOLOv3 model
- **95% detection accuracy** after transfer learning
- **95% tracking accuracy** for proximity risk classification

## Tech Stack

- Python
- OpenCV
- YOLOv3 / SSD MobileNet
- Streamlit

## Project Structure

```
├── YOLO/           # YOLOv3 detection pipeline
├── SSD/            # SSD MobileNet detection pipeline
├── app.py          # Streamlit application entry point
└── README.md
```

## Running the Project

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Author

Aishwarya Ramanath Shanbhag
