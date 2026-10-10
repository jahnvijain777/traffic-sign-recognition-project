# SignSight — Traffic Sign Recognition

SignSight is a web-based traffic sign recognition application that uses machine learning to classify traffic signs from uploaded images. It uses a trained deep learning model to predict the type of traffic sign and display the most likely result along with confidence scores.

The application runs model inference directly in the browser using TensorFlow.js, without requiring a separate backend server for predictions.

## Overview

Traffic sign recognition is an important application of computer vision that helps machines interpret road signs. This project explores how a trained neural network can classify traffic signs into different categories using image processing and deep learning.

SignSight provides a web interface for uploading traffic sign images and viewing the model's predictions.

## Features

- **Image-based recognition:** Upload a traffic sign image for classification.
- **43 traffic sign classes:** Predicts among the classes supported by the trained model.
- **Confidence scores:** View the model's confidence in its predictions.
- **Top predictions:** Compare the most likely alternative classes.
- **Browser-based inference:** Runs predictions locally using TensorFlow.js.
- **Responsive web interface:** Access the application through a modern web browser.

## Tech Stack

| Technology | Purpose |
|---|---|
| React | User interface |
| TypeScript | Type-safe application development |
| Vite | Development server and build tooling |
| TensorFlow.js | Loading the model and running predictions |
| Python / Keras | Model training and model export |
| CSS | Interface styling |

## Dataset

The model was trained using the GTSRB-style traffic sign dataset available in the project.

The dataset contains 43 classes of German traffic signs, including:

- Speed limits
- Stop and yield signs
- Warning signs
- Road restrictions
- Mandatory directions
- Pedestrian and road hazard warnings

Before prediction, input images are resized to **30 × 30 pixels**, converted to RGB pixel values, and normalized to the range of 0–1 to match the model's expected input format.

## How It Works

1. **Upload:** The user selects a traffic sign image.
2. **Preprocessing:** The image is resized and its pixel values are normalized.
3. **Inference:** TensorFlow.js passes the processed image to the trained model.
4. **Classification:** The model generates scores for the 43 supported classes.
5. **Results:** The application displays the predicted sign and alternative predictions with their confidence scores.

## Getting Started

### Prerequisites

Install the following before running the project:

- [Node.js](https://nodejs.org/)
- npm, which is included with Node.js
- Git (optional, for cloning the repository)

### Installation

Clone the repository and navigate into the project directory:

```bash
git clone https://github.com/jahnvijain777/traffic-sign-recognition-project.git
cd traffic-sign-recognition-project
```

Install the dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

Open the local URL displayed in your terminal, usually `http://localhost:5173`.

### Important: Model Files

The application expects the exported TensorFlow.js model at:

```text
public/model/model.json
```

The corresponding model weight files must also be present. Without these files, the web application cannot load the trained model or generate predictions.

If you need to regenerate the browser-compatible model from the original Keras model, use the project's export workflow:

```bash
python -m pip install tensorflowjs

tensorflowjs_converter --input_format=keras archive/training/TSR.h5 public/model

python scripts/normalize_tfjs_model.py
```

Run the export commands from the project root in a Python environment with the required TensorFlow/Keras dependencies installed.

## Building for Production

To create a production build:

```bash
npm run build
```

To preview the production build locally:

```bash
npm run preview
```

The generated production files are placed in the `dist` directory.

## Model Performance

The training notebook reports approximately **99.5% validation accuracy**. This is a validation result, not a guarantee of real-world performance.

Prediction quality can vary depending on image quality, lighting, blur, angle, occlusion, and how closely an image resembles the training data.

## Privacy

Image classification is performed locally in the browser. Uploaded images are not sent to a prediction API by this application.

## Future Improvements

Potential enhancements include:

- Support for Indian traffic sign datasets.
- Testing on a wider variety of real-world road images.
- Improved handling of low-quality and partially obscured images.
- Expanded model evaluation and performance visualizations.
- Deployment of a live demonstration.

## Learning Outcomes

This project explores practical applications of machine learning, computer vision, image preprocessing, model inference, and frontend development. It also demonstrates how a trained deep learning model can be integrated into a browser-based application.

## Author

**Jahnvi Jain**

GitHub: [@jahnvijain777](https://github.com/jahnvijain777)

---

*SignSight — Bringing machine learning and computer vision to traffic sign recognition.*
