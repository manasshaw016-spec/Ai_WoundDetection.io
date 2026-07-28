from __future__ import annotations

import os

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from image_processing import generate_gradcam, overlay_segmentation, to_rgb
from model_inference import analyze_image, load_model
from report_generation import create_pdf_report
from ui import apply_theme, render_sidebar
from utils import HISTORY_CSV, REPORTS_DIR, now_timestamp, read_uploaded_image

st.set_page_config(page_title="AI Wound Support Dashboard", page_icon="🩹", layout="wide")

if "theme" not in st.session_state:
    st.session_state["theme"] = "light"

if "model" not in st.session_state:
    with st.spinner("Loading model..."):
        model, model_type = load_model()
        st.session_state["model"] = model
        st.session_state["model_type"] = model_type

apply_theme(st.session_state["theme"])
page, _ = render_sidebar("Ready")

if page == "History":
    st.title("📈 Prediction History")
    if HISTORY_CSV.exists():
        history_df = pd.read_csv(HISTORY_CSV)
        if not history_df.empty:
            st.subheader("Recent findings")
            st.dataframe(history_df.tail(10), use_container_width=True)
            if "confidence" in history_df.columns:
                fig = px.line(history_df, x="timestamp", y="confidence", markers=True, title="Confidence trend")
                st.plotly_chart(fig, use_container_width=True)
            if {"risk", "severity"}.issubset(history_df.columns):
                risk_counts = history_df["risk"].value_counts()
                severity_counts = history_df["severity"].value_counts()
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(px.bar(risk_counts, title="Risk distribution", labels={"index": "Risk", "value": "Count"}), use_container_width=True)
                with col2:
                    st.plotly_chart(px.bar(severity_counts, title="Severity distribution", labels={"index": "Severity", "value": "Count"}), use_container_width=True)
        else:
            st.info("No predictions have been saved yet.")
    else:
        st.info("No predictions have been saved yet.")
    st.stop()

if page == "About":
    st.title("ℹ️ About this dashboard")
    st.markdown(
        """
        This dashboard is an educational medical-support prototype that combines a local image classifier with OpenCV-based
        wound analysis. It can help highlight suspicious regions, estimate wound area, and organize findings for review.

        Disclaimer: This tool is not a substitute for professional medical advice. It should not be used to prescribe treatment.
        """
    )
    st.stop()

st.title("🩹 AI-Powered Medical Decision Support Dashboard")
st.caption("Upload an image to view wound segmentation, redness analysis, a category estimate, and educational guidance.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"], help="Upload a wound or healthy image to analyze")

if uploaded_file is None:
    st.info("Please upload an image to start the analysis.")
    st.stop()

image = read_uploaded_image(uploaded_file)
if image is None:
    st.error("The uploaded file could not be read. Please try another image.")
    st.stop()

analysis = analyze_image(image, model=st.session_state["model"], model_type=st.session_state["model_type"])
segmentation_mask = analysis["segmentation_mask"]
redness_mask = analysis["redness_mask"]
segmentation_overlay = overlay_segmentation(image, segmentation_mask)
redness_overlay = overlay_segmentation(image, redness_mask, color=(0, 255, 0), alpha=0.35)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Uploaded image")
    st.image(to_rgb(image), caption="Original image", use_container_width=True)
with col2:
    st.subheader("Segmentation overlay")
    st.image(to_rgb(segmentation_overlay), caption="Wound region overlay", use_container_width=True)

st.subheader("Analysis summary")
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
with metric_col1:
    st.metric("Prediction", analysis["label"])
with metric_col2:
    st.metric("Category", analysis["category"])
with metric_col3:
    st.metric("Severity", analysis["severity"])
with metric_col4:
    st.metric("Infection risk", analysis["risk"])

confidence_col1, confidence_col2 = st.columns([2, 1])
with confidence_col1:
    st.write("Confidence")
    st.progress(int(analysis["confidence"]))
    st.caption(f"Estimated confidence: {analysis['confidence']:.1f}%")
with confidence_col2:
    st.metric("Confidence", f"{analysis['confidence']:.1f}%")

st.subheader("Detailed findings")
info_col1, info_col2, info_col3 = st.columns(3)
with info_col1:
    st.metric("Wound area", f"{analysis['wound_area_pixels']} px")
    st.metric("Area percentage", f"{analysis['wound_area_percentage']:.2f}%")
with info_col2:
    st.metric("Redness area", f"{analysis['redness_pixels']} px")
    st.metric("Redness ratio", f"{analysis['redness_ratio']:.2f}%")
with info_col3:
    st.metric("Dominant color", analysis["dominant_color"])
    st.metric("Medical evaluation", "May be advisable" if analysis["medical_evaluation_advisable"] else "Monitor closely")

st.subheader("Redness analysis")
st.image(to_rgb(redness_overlay), caption="Inflamed region highlight", use_container_width=True)

st.markdown("**Color interpretation:**")
st.write(analysis["color_explanation"])

st.subheader("Educational wound-care suggestions")
for suggestion in analysis["educational_guidance"]:
    st.info(suggestion)

st.warning("Disclaimer: This tool is for educational support only and is not a substitute for professional medical advice.")

if st.button("Generate PDF report"):
    report_path = create_pdf_report(os.path.join(REPORTS_DIR, f"{uploaded_file.name.split('.')[0]}_report.pdf"), image, analysis, uploaded_file.name)
    st.success(f"PDF report saved to {report_path}")
    with open(report_path, "rb") as f:
        st.download_button("Download report", f, file_name=os.path.basename(report_path))

st.subheader("Grad-CAM preview")
try:
    gradcam_image = generate_gradcam(st.session_state["model"], image)
    if gradcam_image is not None:
        st.image(to_rgb(gradcam_image), caption="Grad-CAM overlay", use_container_width=True)
    else:
        st.caption("Grad-CAM is not available for the loaded model type.")
except Exception as exc:
    st.caption(f"Grad-CAM preview unavailable: {exc}")

history_entry = {
    "timestamp": now_timestamp(),
    "filename": uploaded_file.name,
    "prediction": analysis["label"],
    "category": analysis["category"],
    "severity": analysis["severity"],
    "risk": analysis["risk"],
    "confidence": analysis["confidence"],
    "wound_area_percentage": analysis["wound_area_percentage"],
    "redness_ratio": analysis["redness_ratio"],
}

if os.path.exists(HISTORY_CSV):
    history_df = pd.read_csv(HISTORY_CSV)
    history_df = pd.concat([history_df, pd.DataFrame([history_entry])], ignore_index=True)
else:
    history_df = pd.DataFrame([history_entry])

history_df.to_csv(HISTORY_CSV, index=False)

st.subheader("Prediction history snapshot")
if not history_df.empty:
    st.dataframe(history_df.tail(5), use_container_width=True)
