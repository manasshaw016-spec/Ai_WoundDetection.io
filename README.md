# 🩹 AI Wound Detection & Decision Support Dashboard

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/Vision-OpenCV-5C3EE8?logo=opencv)](https://opencv.org/)

An interactive, AI-driven medical decision support prototype designed to assist healthcare professionals and researchers in evaluating wound images. The application combines Computer Vision (HSV segmentation, redness mapping, Grad-CAM heatmaps) with Machine Learning classification to provide automated tissue analysis, infection risk estimation, severity scoring, and exportable PDF medical reports.

---

## 🌟 Key Features

- 🔬 **Image Classification**: Classifies uploaded images into **Wound** vs. **Healthy** states using pre-trained Deep Learning CNNs or ML fallback pipelines (Scikit-Learn / Logistic Regression).
- 🎯 **Wound Region Segmentation**: Automatically isolates wound boundaries using adaptive HSV thresholding and Otsu thresholding.
- 🔴 **Redness & Inflammation Detection**: Pinpoints inflamed or reddened skin areas to highlight potential infection markers.
- 🗺️ **Grad-CAM Visualization**: Computes model activation heatmaps to interpret and explain AI decision-making.
- 📊 **Quantitative Metrics**: Measures total wound surface area (pixel count and percentage of frame) and dominant tissue color analysis.
- 📄 **Automated PDF Reports**: Generates downloadable, styled PDF summary reports using ReportLab with embedded images and metric tables.
- 📈 **Prediction History Analytics**: Logs diagnostic predictions to CSV and displays interactive Plotly charts tracking confidence trends, severity distributions, and risk factors over time.
- 🎨 **Customizable Theme**: Supports light and dark mode toggles in the Streamlit UI.

---

## 📁 Repository Structure

```text
Ai_WoundDetection.io/
├── app.py                   # Main Streamlit Dashboard Application
├── main.py                  # Model Training, Dataset Management & Benchmarking
├── model_inference.py       # Inference Pipeline, Grad-CAM & Prediction Logic
├── image_processing.py      # OpenCV Wound Segmentation, Redness & Color Analysis
├── report_generation.py     # ReportLab PDF Report Compiler
├── ui.py                    # Streamlit Layout, Sidebar & Styling Theme
├── utils.py                 # File I/O, Image Helpers & Storage Paths
├── requirements.txt         # Project Dependencies
├── requirements.py          # Python Script for Automated Package Installation
├── wound_model.joblib       # Pre-trained Fallback Machine Learning Model
├── prediction_history.csv   # Prediction History Log
├── data/                    # Dataset Directory (Train / Test split for Wound & Healthy)
│   ├── train/
│   │   ├── wound/
│   │   └── healthy/
│   └── test/
│       ├── wound/
│       └── healthy/
├── reports/                 # Output Folder for Generated PDF Reports
├── LICENSE                  # MIT Open Source License
└── README.md                # Project Documentation
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.8 or higher installed on your machine.
- `pip` package manager.

### 2. Clone the Repository
```bash
git clone https://github.com/manasshaw016-spec/Ai_WoundDetection.io.git
cd Ai_WoundDetection.io
```

### 3. Set Up Virtual Environment (Recommended)
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
Install all required libraries using `requirements.txt`:
```bash
pip install -r requirements.txt
```
*Or run the helper script:*
```bash
python requirements.py
```

---

## 💻 Running the Application

Launch the interactive Streamlit dashboard:

```bash
streamlit run app.py
```

Once launched, open your web browser at `http://localhost:8501`.

### Dashboard Workflow:
1. **Upload Image**: Drag and drop a `.jpg`, `.jpeg`, `.png`, or `.bmp` wound or skin image.
2. **Review Analysis**: View original vs. segmented wound overlays side-by-side.
3. **Inspect Metrics**: Examine predicted classification, confidence score, infection risk, severity rating, and calculated wound area.
4. **Grad-CAM Explanations**: Expand the Grad-CAM section to view visual AI attention regions.
5. **Download PDF Report**: Click **"Generate PDF Report"** to save a detailed summary document locally.
6. **Analytics History**: Navigate to **"History"** from the sidebar to view trend charts and historical prediction logs.

---

## 🏋️ Training the Model

To train or re-evaluate the model on custom datasets:

1. Place image files into the appropriate folders under `data/train/` and `data/test/`:
   - `data/train/wound/`
   - `data/train/healthy/`
   - `data/test/wound/`
   - `data/test/healthy/`
2. Run the main training script:
   ```bash
   python main.py
   ```

---

## ⚠️ Disclaimer

> [!IMPORTANT]
> **Educational & Decision Support Only**: This tool is designed strictly for research, educational prototyping, and preliminary decision-support purposes. It is **not** a certified medical diagnostic device and must **not** be used as a substitute for professional clinical judgment, diagnosis, or treatment planning.

---

## 📄 License

Distributed under the [MIT License](LICENSE). See `LICENSE` for more information.

---

### 👨‍💻 Created By
**[Manas Kumar Shaw](https://github.com/manasshaw016-spec)**
