import subprocess
import sys

REQUIRED_PACKAGES = [
    "numpy",
    "opencv-python",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "streamlit",
    "pandas",
    "plotly",
    "pillow",
    "reportlab",
]


def install_requirements():
    for package in REQUIRED_PACKAGES:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])


if __name__ == "__main__":
    install_requirements()
