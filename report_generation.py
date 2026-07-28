from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def create_pdf_report(output_path, image_bgr, analysis, filename):
    """Generate a lightweight PDF report for the current prediction."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("AI Wound Detection Report", styles["Title"]))
    story.append(Paragraph("Educational support only — not a medical diagnosis.", styles["BodyText"]))
    story.append(Spacer(1, 0.15 * inch))

    if image_bgr is not None:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        _, png_bytes = cv2.imencode(".png", image_rgb)
        image_obj = Image(io.BytesIO(png_bytes.tobytes()), width=2.5 * inch, height=2.5 * inch)
        story.append(image_obj)
        story.append(Spacer(1, 0.15 * inch))

    data = [
        ["Field", "Value"],
        ["File", filename],
        ["Prediction", analysis.get("label", "Unknown")],
        ["Category", analysis.get("category", "Unknown")],
        ["Confidence", f"{analysis.get('confidence', 0):.1f}%"],
        ["Wound Area", f"{analysis.get('wound_area_pixels', 0)} px ({analysis.get('wound_area_percentage', 0):.2f}%)"],
        ["Redness", f"{analysis.get('redness_ratio', 0):.2f}%"],
        ["Severity", analysis.get("severity", "Unknown")],
        ["Infection risk", analysis.get("risk", "Unknown")],
        ["Dominant color", analysis.get("dominant_color", "Unknown")],
    ]

    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563eb")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Educational guidance", styles["Heading2"]))
    for guidance in analysis.get("educational_guidance", []):
        story.append(Paragraph(f"• {guidance}", styles["BodyText"]))

    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("Disclaimer: This tool is for educational support only and is not a substitute for professional medical advice.", styles["BodyText"]))

    doc.build(story)
    return str(output_path)
