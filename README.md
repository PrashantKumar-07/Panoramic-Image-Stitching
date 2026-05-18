# Panoramic Image Stitching: Production-Grade Implementation

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/opencv-%3E%3D4.0-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

> A production-grade panoramic image stitching engine built from first principles. Implements state-of-the-art computer vision algorithms (SIFT, RANSAC, homography estimation) to produce seamless, multi-perspective panoramas with sub-pixel accuracy.

![Panorama Result](panorama.jpg)

---

## Overview

This project tackles the challenge of image stitching by solving multiple interconnected computer vision problems, including descriptor space alignment, geometric transformation, continuous domain mapping, and multi-image fusion.

### Core Capabilities

- **SIFT Feature Detection:** Scale & rotation invariant keypoint extraction
- **Robust Feature Matching:** Lowe's ratio test for discriminative correspondence
- **Homography Estimation:** DLT + RANSAC for perspective transformation
- **Advanced Image Warping:** Bilinear interpolation for anti-aliased output
- **Exposure Compensation:** Automatic brightness harmonization
- **Seamless Blending:** Distance-weighted feathering for invisible seams

---

## Project Structure

```text
CV Assignment_(M25CSE023)/
├── README.md                           # Documentation
├── main.py                             # Python implementation script
├── CV_assignment(M25CSE023).ipynb      # Interactive Jupyter notebook
├── CV_report_(M25CSE023).pdf           # Detailed Assignment Report
├── left.jpg                            # Input: leftmost image
├── center.jpg                          # Input: center reference
├── right.jpg                           # Input: rightmost image
└── panorama.jpg                        # Output: stitched result
```

---

## Quick Start

### Requirements

- Python 3.8+
- `opencv-python` >= 4.0
- `numpy` >= 1.19
- `matplotlib` >= 3.3

### Installation & Usage

```bash
# Clone the repository
git clone https://github.com/PrashantKumar-07/CV-Assignment-M25CSE023.git
cd "CV-Assignment-M25CSE023"

# Install dependencies
pip install opencv-python numpy matplotlib

# Execute the script
python main.py
```

*Alternatively, you can run the interactive Jupyter notebook `CV_assignment(M25CSE023).ipynb`.*

### Expected Output

```text
Saving panorama.jpg...
✓ Output Saved: panorama.jpg
```

---

## Results & Validation

The pipeline successfully aligns and stitches three overlapping images into a single cohesive panorama.

- **Imperceptible Seams:** Distance-weighted blending ensures smooth transitions.
- **Color Consistency:** Exposure compensation corrects brightness variations across the images.
- **Geometric Accuracy:** RANSAC reliably filters out mismatched features to produce an accurate homography matrix.

For an in-depth explanation of the mathematical foundations and detailed benchmark results, please refer to the attached `CV_report_(M25CSE023).pdf`.

---

## References

1. D. G. Lowe, "Distinctive Image Features from Scale-Invariant Keypoints," *IJCV*, 2004
2. M. A. Fischler & R. C. Bolles, "Random Sample Consensus," *CACM*, 1981
3. R. Hartley & A. Zisserman, *Multiple View Geometry in Computer Vision*, 2003
