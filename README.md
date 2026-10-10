# SignSight

A client-side traffic sign recognition web app powered by the trained 43-class Keras model in this project. The browser resizes an uploaded image to `30x30`, normalizes RGB values to `0..1`, and runs inference locally with TensorFlow.js.

## Run locally

```bash
npm install
npm run dev
```

The first model export creates `public/model/model.json` and its weight shard(s). The exported files are required for the app to show predictions.

## Model export

The original model is also available as `archive/training/TSR.h5`, which is the portable input expected by the TensorFlow.js converter. To regenerate the browser model, install the converter in the Python environment that has TensorFlow/Keras, then run:

```bash
python -m pip install tensorflowjs
tensorflowjs_converter --input_format=keras archive/training/TSR.h5 public/model
python scripts/normalize_tfjs_model.py
```

The normalization command adapts the Keras 3 metadata and weight names for the browser LayersModel loader.

On Windows, use the Python executable from your environment instead of `python` if necessary. The web app does not need Python at runtime.

## Deploy to Vercel

Import this repository into Vercel with the default Vite settings:

- Build command: `npm run build`
- Output directory: `dist`
- Install command: `npm install`

No environment variables or server-side model hosting are needed. Inference is performed in the visitor's browser, so uploaded images are not sent to an API.

## Model notes

The model was trained on the GTSRB-style dataset included under `archive/Train`, using 43 classes and `30x30` RGB images. The notebook reports approximately 99.5% validation accuracy, but real-world images can be less reliable when signs are blurry, occluded, or poorly lit.
