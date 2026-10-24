
import React from "react";
import ReactDOM from "react-dom/client";
import * as tf from "@tensorflow/tfjs";
import "./styles.css";

const SIGN_NAMES = [
  
  "Speed limit (20km/h)",
  "Speed limit (30km/h)",
  "Speed limit (50km/h)",
  "Speed limit (60km/h)",
  "Speed limit (70km/h)",
  "Speed limit (80km/h)",
  "End of speed limit (80km/h)",
  "Speed limit (100km/h)",
  "Speed limit (120km/h)",
  "No passing",
  "No passing vehicles over 3.5 tons",
  "Right-of-way at intersection",
  "Priority road",
  "Yield",
  "Stop",
  "No vehicles",
  "Vehicles over 3.5 tons prohibited",
  "No entry",
  "General caution",
  "Dangerous curve left",
  "Dangerous curve right",
  "Double curve",
  "Bumpy road",
  "Slippery road",
  "Road narrows on the right",
  "Road work",
  "Traffic signals",
  "Pedestrians",
  "Children crossing",
  "Bicycles crossing",
  "Beware of ice/snow",
  "Wild animals crossing",
  "End of speed + passing limits",
  "Turn right ahead",
  "Turn left ahead",
  "Ahead only",
  "Go straight or right",
  "Go straight or left",
  "Keep right",
  "Keep left",
  "Roundabout mandatory",
  "End of no passing",
  "End of no passing for vehicles over 3.5 tons",
];

type Prediction = {
  index: number;
  name: string;
  confidence: number;
};

function App() {
  const [model, setModel] = React.useState<tf.LayersModel | null>(null);
  const [imageUrl, setImageUrl] = React.useState<string | null>(null);
  const [prediction, setPrediction] = React.useState<Prediction | null>(null);
  const [alternatives, setAlternatives] = React.useState<Prediction[]>([]);
  const [loading, setLoading] = React.useState(false);
  const [status, setStatus] = React.useState(
    "Loading recognition model…"
  );

  const inputRef = React.useRef<HTMLInputElement>(null);

  // LOAD ML MODEL
  React.useEffect(() => {
    let active = true;

    tf.loadLayersModel("/model/model.json")
      .then((loaded) => {
        if (active) {
          setModel(loaded);
          setStatus(
            "Model ready · predictions run locally in your browser"
          );
        }
      })
      .catch((error: unknown) => {
        console.error("Traffic sign model failed to load", error);

        if (active) {
          setStatus(
            "Could not load the model. Check that public/model is exported."
          );
        }
      });

    return () => {
      active = false;
    };
  }, []);

  // IMAGE CLASSIFICATION
  const classify = async (file: File) => {
    if (!model) return;

    const url = URL.createObjectURL(file);

    setImageUrl((oldUrl) => {
      if (oldUrl) URL.revokeObjectURL(oldUrl);
      return url;
    });

    setPrediction(null);
    setAlternatives([]);
    setLoading(true);

    try {
      const image = new Image();
      image.src = url;

      await image.decode();

      const values = tf.tidy(() => {
        const input = tf.browser
          .fromPixels(image)
          .resizeBilinear([30, 30])
          .toFloat()
          .div(255)
          .expandDims(0);

        return model.predict(input) as tf.Tensor;
      });

      const scores = Array.from(await values.data());
      values.dispose();

      const ranked = scores
        .map((confidence, index) => ({
          index,
          name: SIGN_NAMES[index],
          confidence,
        }))
        .sort((a, b) => b.confidence - a.confidence);

      setPrediction(ranked[0]);
      setAlternatives(ranked.slice(1, 4));
    } catch {
      setStatus(
        "That image could not be processed. Please try a PNG or JPEG."
      );
    } finally {
      setLoading(false);
    }
  };

  const onFile = (file?: File) => {
    if (file && file.type.startsWith("image/")) {
      void classify(file);
    }
  };

  return (
    <main className="shell">

      {/* NAVIGATION */}

      <nav className="nav">
        <div className="brand">
          <span className="brand-mark">◈</span> SIGNSIGHT
        </div>

        <div className="nav-links">
          <a href="#home">Home</a>
          <a href="#detect">Detection</a>
          <a href="#about">About</a>
        </div>

        <a href="#detect" className="nav-button">
          Launch Detection →
        </a>
      </nav>

      {/* HERO SECTION */}

      <section className="hero" id="home">
        <div className="eyebrow">
          COMPUTER VISION / TRAFFIC INTELLIGENCE
        </div>

        <h1>
          Making Roads <br />
          Smarter Through <em>Vision.</em>
        </h1>

        <p className="intro">
          SignSight is a deep learning-powered traffic sign
          recognition platform designed to identify and classify
          road signs through intelligent image analysis.
        </p>

        <div className="hero-actions">
          <a href="#detect" className="primary-button">
            Explore Detection →
          </a>

          <a href="#about" className="secondary-button">
            About the Technology
          </a>
        </div>
      </section>

      {/* DETECTION SECTION */}

      <section className="workspace" id="detect">

        <div
          className={`dropzone ${imageUrl ? "has-image" : ""}`}
          onClick={() => inputRef.current?.click()}
          onDragOver={(event) => event.preventDefault()}
          onDrop={(event) => {
            event.preventDefault();
            onFile(event.dataTransfer.files[0]);
          }}
          role="button"
          tabIndex={0}
          onKeyDown={(event) => {
            if (event.key === "Enter" || event.key === " ") {
              inputRef.current?.click();
            }
          }}
        >
          {imageUrl ? (
            <img
              src={imageUrl}
              alt="Uploaded traffic sign"
            />
          ) : (
            <div className="drop-content">
              <div className="upload-icon">↑</div>

              <strong>Drop a sign photo here</strong>

              <span>
                or click to browse · PNG, JPG or WebP
              </span>
            </div>
          )}

          <input
            ref={inputRef}
            type="file"
            accept="image/png,image/jpeg,image/webp"
            hidden
            onChange={(event) =>
              onFile(event.target.files?.[0])
            }
          />
        </div>

        {/* PREDICTION RESULT */}

        <div className="result-card">
          <div className="card-label">
            RECOGNITION RESULT
          </div>

          {loading ? (
            <div className="result-placeholder">
              <span className="spinner" />
              Reading sign…
            </div>
          ) : prediction ? (
            <div className="result">
              <div className="result-index">
                {String(prediction.index).padStart(2, "0")}
              </div>

              <div className="result-copy">
                <span className="confidence">
                  {Math.round(prediction.confidence * 100)}%
                  confidence
                </span>

                <h2>{prediction.name}</h2>
              </div>
            </div>
          ) : (
            <div className="result-placeholder">
              Upload an image to see the prediction
            </div>
          )}

          {alternatives.length > 0 && (
            <div className="alternatives">
              <span>Other possibilities</span>

              {alternatives.map((item) => (
                <div
                  className="alternative"
                  key={item.index}
                >
                  <b>{item.name}</b>

                  <span>
                    {Math.round(item.confidence * 100)}%
                  </span>
                </div>
              ))}
            </div>
          )}

          <div className="model-status">
            <span
              className={
                model ? "status-dot ready" : "status-dot"
              }
            />

            {status}
          </div>
        </div>
      </section>

      {/* ABOUT SECTION */}

      <section className="about-section" id="about">
        <div className="eyebrow">
          BEHIND THE TECHNOLOGY
        </div>

        <h2>
          Intelligence Behind Every Sign.
        </h2>

        <p>
          SignSight uses Deep Learning and Computer Vision to
          classify traffic signs from uploaded images. The system
          processes images through a trained neural network and
          displays the predicted sign along with its confidence score.
        </p>

        <div className="about-stats">
          <div>
            <strong>43</strong>
            <span>Traffic Sign Classes</span>
          </div>

          <div>
            <strong>30 × 30</strong>
            <span>Image Input Size</span>
          </div>

          <div>
            <strong>TensorFlow.js</strong>
            <span>Browser-Based Inference</span>
          </div>
        </div>
      </section>

      {/* FOOTER */}

      <footer>
        <span>
          43 sign classes · 30×30 RGB input
        </span>

        <span>
          Built with TensorFlow.js · runs client-side
        </span>
      </footer>

    </main>
  );
}

ReactDOM.createRoot(
  document.getElementById("root")!
).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
