
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

type Crop = {
  x: number;
  y: number;
  w: number;
  h: number;
};

const DEFAULT_CROP: Crop = {
  x: 0.2,
  y: 0.2,
  w: 0.6,
  h: 0.6,
};

function App() {
  // MODEL

  const [model, setModel] =
    React.useState<tf.LayersModel | null>(null);

  const [status, setStatus] = React.useState(
    "Loading recognition model…"
  );

  // IMAGE UPLOAD

  const inputRef = React.useRef<HTMLInputElement>(null);

  const [imageUrl, setImageUrl] =
    React.useState<string | null>(null);

  const [prediction, setPrediction] =
    React.useState<Prediction | null>(null);

  const [alternatives, setAlternatives] =
    React.useState<Prediction[]>([]);

  const [loading, setLoading] = React.useState(false);

  // CAMERA

  const videoRef = React.useRef<HTMLVideoElement>(null);
  const streamRef = React.useRef<MediaStream | null>(null);
  const cameraSessionRef = React.useRef(0);

  const [cameraActive, setCameraActive] =
    React.useState(false);

  const [cameraStarting, setCameraStarting] =
    React.useState(false);

  const [cameraProcessing, setCameraProcessing] =
    React.useState(false);

  const [capturedPhoto, setCapturedPhoto] =
    React.useState<string | null>(null);

  const [cameraPrediction, setCameraPrediction] =
    React.useState<Prediction | null>(null);

  const [cameraAlternatives, setCameraAlternatives] =
    React.useState<Prediction[]>([]);

  const [cameraStatus, setCameraStatus] = React.useState(
    "Camera is currently off."
  );

  // CROPPING

  const cropImageRef = React.useRef<HTMLImageElement>(null);

  const cropDragRef = React.useRef<{
    x: number;
    y: number;
  } | null>(null);

  const [crop, setCrop] =
    React.useState<Crop>(DEFAULT_CROP);

  // LOAD MODEL

  React.useEffect(() => {
    let active = true;
    let loadedModel: tf.LayersModel | null = null;

    tf.loadLayersModel("/model/model.json")
      .then((loaded) => {
        if (!active) {
          loaded.dispose();
          return;
        }

        loadedModel = loaded;
        setModel(loaded);

        setStatus(
          "Model ready · predictions run locally in your browser"
        );
      })
      .catch((error: unknown) => {
        console.error("Model loading failed:", error);

        if (active) {
          setStatus(
            "Could not load model. Check public/model/model.json."
          );
        }
      });

    return () => {
      active = false;
      loadedModel?.dispose();
    };
  }, []);

  // SHARED PREDICTION FUNCTION

  const predictSign = async (
    source: HTMLImageElement,
    currentModel: tf.LayersModel
  ): Promise<Prediction[]> => {
    const output = tf.tidy(() => {
      const input = tf.browser
        .fromPixels(source)
        .resizeBilinear([30, 30])
        .toFloat()
        .div(255)
        .expandDims(0);

      return currentModel.predict(input) as tf.Tensor;
    });

    try {
      const scores = Array.from(await output.data());

      return scores
        .map((confidence, index) => ({
          index,
          name: SIGN_NAMES[index] ?? `Class ${index}`,
          confidence,
        }))
        .sort((a, b) => b.confidence - a.confidence);
    } finally {
      output.dispose();
    }
  };

  // IMAGE UPLOAD CLASSIFICATION

  const classify = async (file: File) => {
    if (!model || loading) return;

    const url = URL.createObjectURL(file);

    setImageUrl((previous) => {
      if (previous) URL.revokeObjectURL(previous);
      return url;
    });

    setPrediction(null);
    setAlternatives([]);
    setLoading(true);

    try {
      const image = new Image();
      image.src = url;

      await image.decode();

      const ranked = await predictSign(image, model);

      setPrediction(ranked[0]);
      setAlternatives(ranked.slice(1, 4));

      setStatus("Image recognized successfully.");
    } catch (error) {
      console.error(error);
      setStatus("Unable to process this image.");
    } finally {
      setLoading(false);
    }
  };

  const onFile = (file?: File) => {
    if (!file) return;

    if (!file.type.startsWith("image/")) {
      setStatus("Please select a valid image.");
      return;
    }

    void classify(file);
  };

  // RELEASE CAMERA

  const releaseCamera = React.useCallback(() => {
    streamRef.current
      ?.getTracks()
      .forEach((track) => track.stop());

    streamRef.current = null;

    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject = null;
    }

    setCameraActive(false);
  }, []);

  // STOP CAMERA AND RESET

  const stopCamera = React.useCallback(() => {
    cameraSessionRef.current += 1;

    releaseCamera();

    setCameraStarting(false);
    setCameraProcessing(false);
    setCapturedPhoto(null);
    setCameraPrediction(null);
    setCameraAlternatives([]);
    setCrop(DEFAULT_CROP);

    setCameraStatus("Camera is currently off.");
  }, [releaseCamera]);

  // START CAMERA

  const startCamera = async () => {
    if (
      !model ||
      cameraActive ||
      cameraStarting ||
      cameraProcessing
    ) {
      return;
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraStatus(
        "Camera requires localhost or a secure HTTPS website."
      );
      return;
    }

    setCameraStarting(true);
    setCapturedPhoto(null);
    setCameraPrediction(null);
    setCameraAlternatives([]);
    setCrop(DEFAULT_CROP);

    setCameraStatus("Requesting camera permission…");

    const session = ++cameraSessionRef.current;

    let stream: MediaStream | null = null;

    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: "environment",
        },
      });

      if (session !== cameraSessionRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }

      const video = videoRef.current;

      if (!video) {
        throw new Error("Video element unavailable.");
      }

      video.srcObject = stream;
      streamRef.current = stream;

      await video.play();

      if (session !== cameraSessionRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }

      setCameraActive(true);

      setCameraStatus(
        "Camera ready. Position the sign and click Capture Photo."
      );
    } catch (error) {
      console.error("Camera error:", error);

      stream?.getTracks().forEach((track) => track.stop());

      if (session === cameraSessionRef.current) {
        releaseCamera();

        setCameraStatus(
          "Camera unavailable. Check Safari camera permissions."
        );
      }
    } finally {
      if (session === cameraSessionRef.current) {
        setCameraStarting(false);
      }
    }
  };

  // CAPTURE PHOTO - NO AUTOMATIC PREDICTION

  const capturePhoto = () => {
    const video = videoRef.current;

    if (
      !cameraActive ||
      !video ||
      cameraProcessing ||
      video.readyState <
        HTMLMediaElement.HAVE_CURRENT_DATA
    ) {
      return;
    }

    try {
      const canvas = document.createElement("canvas");

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      if (!canvas.width || !canvas.height) {
        throw new Error("Camera frame is not ready.");
      }

      const context = canvas.getContext("2d");

      if (!context) {
        throw new Error("Cannot create canvas.");
      }

      context.drawImage(
        video,
        0,
        0,
        canvas.width,
        canvas.height
      );

      const photo = canvas.toDataURL("image/jpeg", 0.92);

      releaseCamera();

      setCapturedPhoto(photo);
      setCrop(DEFAULT_CROP);
      setCameraPrediction(null);
      setCameraAlternatives([]);

      setCameraStatus(
        "Photo captured! Drag a box around ONE traffic sign, then click Recognize Selected Sign."
      );
    } catch (error) {
      console.error(error);

      setCameraStatus(
        "Could not capture photo. Please try again."
      );
    }
  };

  // CONVERT MOUSE / TOUCH POSITION TO IMAGE COORDINATES

  const getCropPoint = (
    event: React.PointerEvent<HTMLDivElement>
  ) => {
    const image = cropImageRef.current;

    if (!image) return null;

    const rect = image.getBoundingClientRect();

    if (!rect.width || !rect.height) return null;

    return {
      x: Math.max(
        0,
        Math.min(
          1,
          (event.clientX - rect.left) / rect.width
        )
      ),
      y: Math.max(
        0,
        Math.min(
          1,
          (event.clientY - rect.top) / rect.height
        )
      ),
    };
  };

  // START DRAWING CROP

  const beginCrop = (
    event: React.PointerEvent<HTMLDivElement>
  ) => {
    if (cameraProcessing) return;

    const point = getCropPoint(event);

    if (!point) return;

    cropDragRef.current = point;

    event.currentTarget.setPointerCapture(event.pointerId);

    setCrop({
      x: point.x,
      y: point.y,
      w: 0,
      h: 0,
    });

    setCameraPrediction(null);
    setCameraAlternatives([]);
  };

  // UPDATE CROP WHILE DRAGGING

  const moveCrop = (
    event: React.PointerEvent<HTMLDivElement>
  ) => {
    const start = cropDragRef.current;
    const point = getCropPoint(event);

    if (!start || !point) return;

    setCrop({
      x: Math.min(start.x, point.x),
      y: Math.min(start.y, point.y),
      w: Math.abs(point.x - start.x),
      h: Math.abs(point.y - start.y),
    });
  };

  // FINISH CROP

  const endCrop = () => {
    cropDragRef.current = null;
  };

  // RECOGNIZE ONLY THE CROPPED SIGN

  const recognizeCrop = async () => {
    const image = cropImageRef.current;

    if (!image || !model || cameraProcessing) return;

    if (crop.w < 0.025 || crop.h < 0.025) {
      setCameraStatus(
        "Please select a larger area around the traffic sign."
      );
      return;
    }

    const session = cameraSessionRef.current;

    setCameraProcessing(true);
    setCameraPrediction(null);
    setCameraAlternatives([]);

    setCameraStatus("Recognizing selected traffic sign…");

    try {
      const canvas = document.createElement("canvas");

      const sourceX = Math.round(
        crop.x * image.naturalWidth
      );

      const sourceY = Math.round(
        crop.y * image.naturalHeight
      );

      const sourceWidth = Math.max(
        1,
        Math.round(crop.w * image.naturalWidth)
      );

      const sourceHeight = Math.max(
        1,
        Math.round(crop.h * image.naturalHeight)
      );

      canvas.width = sourceWidth;
      canvas.height = sourceHeight;

      const context = canvas.getContext("2d");

      if (!context) {
        throw new Error("Crop canvas unavailable.");
      }

      context.drawImage(
        image,
        sourceX,
        sourceY,
        sourceWidth,
        sourceHeight,
        0,
        0,
        sourceWidth,
        sourceHeight
      );

      const selectedImage = new Image();

      selectedImage.src = canvas.toDataURL("image/png");

      await selectedImage.decode();

      const ranked = await predictSign(
        selectedImage,
        model
      );

      if (session === cameraSessionRef.current) {
        setCameraPrediction(ranked[0]);
        setCameraAlternatives(ranked.slice(1, 4));

        setCameraStatus(
          "Selected sign analyzed. Adjust the crop to test again, or retake the photo."
        );
      }
    } catch (error) {
      console.error("Crop prediction failed:", error);

      if (session === cameraSessionRef.current) {
        setCameraStatus(
          "Unable to recognize the selected area. Try another crop."
        );
      }
    } finally {
      if (session === cameraSessionRef.current) {
        setCameraProcessing(false);
      }
    }
  };

  // CLEAN UP CAMERA WHEN WEBSITE CLOSES

  React.useEffect(() => {
    return () => {
      cameraSessionRef.current += 1;

      streamRef.current
        ?.getTracks()
        .forEach((track) => track.stop());
    };
  }, []);

  // WEBSITE

  return (
    <main className="shell">

      {/* NAVIGATION */}

      <nav className="nav">
        <div className="brand">
          <span className="brand-mark">◈</span>
          {" "}SIGNSIGHT
        </div>

        <div className="nav-links">
          <a href="#home">Home</a>
          <a href="#dashboard">Dashboard</a>
          <a href="#detect">Detection</a>
          <a href="#live-camera">Camera Capture</a>
          <a href="#how-it-works">How It Works</a>
          <a href="#classes">Sign Classes</a>
          <a href="#about">About</a>
        </div>

        <a href="#detect" className="nav-button">
          Launch Detection →
        </a>
      </nav>

      {/* HERO */}

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

          <a href="#live-camera" className="secondary-button">
            Try Camera Capture
          </a>
        </div>
      </section>

      {/* PHASE 7 DASHBOARD */}

      <section className="dashboard-header" id="dashboard">
        <div className="dashboard-topline">
          <span className="eyebrow">
            SIGNSIGHT / AI WORKSPACE
          </span>

          <span className="dashboard-version">
            PLATFORM OVERVIEW / V1.0
          </span>
        </div>

        <h2>
          Recognition Dashboard<span>.</span>
        </h2>

        <p>
          Your intelligent workspace for traffic sign
          classification and camera-based computer vision.
        </p>

        <div className="dashboard-stats">
          <div className="dashboard-stat">
            <span className="dashboard-stat-label">
              SUPPORTED CLASSES
            </span>
            <strong>43</strong>
            <small>Traffic sign classes supported</small>
          </div>

          <div className="dashboard-stat">
            <span className="dashboard-stat-label">
              IMAGE INPUT
            </span>
            <strong>30 × 30</strong>
            <small>RGB neural network input</small>
          </div>

          <div className="dashboard-stat">
            <span className="dashboard-stat-label">
              RECOGNITION MODES
            </span>
            <strong>02</strong>
            <small>Image upload and camera capture</small>
          </div>
        </div>
      </section>

      {/* IMAGE UPLOAD DETECTION */}

      <section className="detection-section" id="detect">
        <div className="detection-heading">
          <div className="eyebrow">
            AI-POWERED RECOGNITION
          </div>

          <h2>Traffic Sign Recognition</h2>

          <p>
            Upload a traffic sign image to explore how
            SignSight classifies road signs using deep learning.
          </p>
        </div>

        <div className="workspace">
          <div
            className={`dropzone ${
              imageUrl ? "has-image" : ""
            }`}
            onClick={() => {
              if (!loading) inputRef.current?.click();
            }}
            onDragOver={(event) => event.preventDefault()}
            onDrop={(event) => {
              event.preventDefault();
              onFile(event.dataTransfer.files[0]);
            }}
            role="button"
            tabIndex={0}
            aria-label="Upload traffic sign image"
            onKeyDown={(event) => {
              if (
                !loading &&
                (event.key === "Enter" || event.key === " ")
              ) {
                event.preventDefault();
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
              onChange={(event) => {
                onFile(event.target.files?.[0]);
                event.target.value = "";
              }}
            />
          </div>

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
                    {Math.round(
                      prediction.confidence * 100
                    )}% confidence
                  </span>

                  <h2>{prediction.name}</h2>

                  <div
                    className="confidence-track"
                    role="progressbar"
                    aria-label="Prediction confidence"
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-valuenow={Math.round(
                      prediction.confidence * 100
                    )}
                  >
                    <div
                      className="confidence-fill"
                      style={{
                        width: `${
                          Math.max(
                            0,
                            Math.min(
                              100,
                              prediction.confidence * 100
                            )
                          )
                        }%`,
                      }}
                    />
                  </div>
                </div>
              </div>
            ) : (
              <div className="result-placeholder">
                Upload an image to see the prediction
              </div>
            )}

            {prediction && !loading && (
              <button
                type="button"
                className="upload-again-button"
                onClick={() => inputRef.current?.click()}
              >
                ↑ Upload Another Image
              </button>
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
        </div>
      </section>

      {/* CAMERA WITH CROPPING */}

      <section className="camera-section" id="live-camera">
        <div className="eyebrow">
          CAMERA-BASED COMPUTER VISION
        </div>

        <h2>Capture, Crop & Recognize.</h2>

        <p className="camera-intro">
          Capture a photo, drag a yellow box around one traffic
          sign, and recognize only the selected area.
        </p>

        <div className="camera-workspace">
          <div className="camera-preview-card">
            <div className="card-label">
              CAMERA / PHOTO CROPPING
            </div>

            <div className="camera-preview">

              {/* LIVE PREVIEW */}

              <video
                ref={videoRef}
                autoPlay
                muted
                playsInline
                aria-label="Camera preview"
                style={{
                  display: cameraActive ? "block" : "none",
                }}
              />

              {/* CAPTURED PHOTO AND CROP BOX */}

              {!cameraActive && capturedPhoto && (
                <div className="crop-photo-wrap">
                  <img
                    ref={cropImageRef}
                    src={capturedPhoto}
                    alt="Captured photo for cropping"
                    draggable={false}
                  />

                  <div
                    className="crop-interaction"
                    onPointerDown={beginCrop}
                    onPointerMove={moveCrop}
                    onPointerUp={endCrop}
                    onPointerCancel={endCrop}
                    aria-label="Drag to select traffic sign"
                  >
                    <div
                      className="crop-selection"
                      style={{
                        left: `${crop.x * 100}%`,
                        top: `${crop.y * 100}%`,
                        width: `${crop.w * 100}%`,
                        height: `${crop.h * 100}%`,
                      }}
                    />
                  </div>
                </div>
              )}

              {/* CAMERA OFF */}

              {!cameraActive && !capturedPhoto && (
                <div className="camera-placeholder">
                  <div className="camera-placeholder-icon">
                    ◉
                  </div>

                  <strong>
                    {cameraStarting
                      ? "Starting camera…"
                      : "Camera is off"}
                  </strong>

                  <span>
                    Open the camera to capture a traffic sign.
                  </span>
                </div>
              )}

              {cameraActive && (
                <div
                  className="camera-guide"
                  aria-hidden="true"
                >
                  <span />
                </div>
              )}
            </div>

            {/* CAMERA CONTROLS */}

            <div className="camera-controls">
              {cameraActive ? (
                <>
                  <button
                    type="button"
                    className="primary-button"
                    onClick={capturePhoto}
                    disabled={cameraProcessing}
                  >
                    ◉ Capture Photo
                  </button>

                  <button
                    type="button"
                    className="camera-stop-button"
                    onClick={stopCamera}
                  >
                    ■ Stop Camera
                  </button>
                </>
              ) : capturedPhoto ? (
                <>
                  <button
                    type="button"
                    className="primary-button"
                    onClick={() => void recognizeCrop()}
                    disabled={
                      !model ||
                      cameraProcessing ||
                      crop.w < 0.025 ||
                      crop.h < 0.025
                    }
                  >
                    {cameraProcessing
                      ? "Recognizing…"
                      : "◎ Recognize Selected Sign"}
                  </button>

                  <button
                    type="button"
                    className="camera-stop-button"
                    onClick={() => void startCamera()}
                    disabled={
                      !model ||
                      cameraProcessing ||
                      cameraStarting
                    }
                  >
                    ↻ Retake Photo
                  </button>
                </>
              ) : (
                <button
                  type="button"
                  className="primary-button"
                  onClick={() => void startCamera()}
                  disabled={!model || cameraStarting}
                >
                  {cameraStarting
                    ? "Starting Camera…"
                    : "◉ Start Camera"}
                </button>
              )}
            </div>

            <p className="camera-status" role="status">
              {cameraStatus}
            </p>
          </div>

          {/* CAMERA RECOGNITION RESULT */}

          <div className="camera-result-card">
            <div className="card-label">
              CROPPED PHOTO RESULT
            </div>

            {cameraPrediction ? (
              <div className="camera-result">
                <span className="camera-result-label">
                  PREDICTED TRAFFIC SIGN
                </span>

                <h3>{cameraPrediction.name}</h3>

                <strong className="camera-confidence">
                  {Math.round(
                    cameraPrediction.confidence * 100
                  )}% confidence
                </strong>

                <div
                  className="confidence-track"
                  role="progressbar"
                  aria-label="Camera prediction confidence"
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={Math.round(
                    cameraPrediction.confidence * 100
                  )}
                >
                  <div
                    className="confidence-fill"
                    style={{
                      width: `${
                        Math.max(
                          0,
                          Math.min(
                            100,
                            cameraPrediction.confidence * 100
                          )
                        )
                      }%`,
                    }}
                  />
                </div>

                {cameraAlternatives.length > 0 && (
                  <div className="alternatives">
                    <span>Other possibilities</span>

                    {cameraAlternatives.map((item) => (
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

                <p className="camera-note">
                  Only your selected crop was classified.
                  Confidence does not guarantee accuracy.
                </p>
              </div>
            ) : (
              <div className="camera-result-empty">
                {cameraProcessing
                  ? "Recognizing selected sign…"
                  : capturedPhoto
                    ? "Drag a box around one sign, then click Recognize Selected Sign."
                    : "Capture a photo to begin."}
              </div>
            )}
          </div>
        </div>
      </section>

      {/* HOW IT WORKS */}

      <section className="how-section" id="how-it-works">
        <div className="eyebrow">
          THE RECOGNITION PROCESS
        </div>

        <h2>How SignSight Works.</h2>

        <p className="how-intro">
          From image input to intelligent classification,
          explore how SignSight identifies traffic signs
          in three simple steps.
        </p>

        <div className="how-grid">
          <div className="how-card">
            <span className="how-number">01 / INPUT</span>
            <div className="how-icon">↑</div>
            <h3>Provide Image</h3>
            <p>
              Upload a traffic sign image or capture and crop
              one using your camera.
            </p>
          </div>

          <div className="how-card">
            <span className="how-number">02 / PROCESS</span>
            <div className="how-icon">◈</div>
            <h3>AI Classification</h3>
            <p>
              The trained neural network processes the image
              and evaluates 43 traffic sign classes.
            </p>
          </div>

          <div className="how-card">
            <span className="how-number">03 / OUTPUT</span>
            <div className="how-icon">✓</div>
            <h3>Recognition Result</h3>
            <p>
              View the predicted sign, its confidence score,
              and other possibilities.
            </p>
          </div>
        </div>
      </section>

      {/* SIGN CLASSES */}

      <section className="classes-section" id="classes">
        <div className="eyebrow">
          SUPPORTED SIGN CATEGORIES
        </div>

        <h2>Understanding Every Sign.</h2>

        <p className="classes-intro">
          SignSight recognises 43 traffic sign classes across
          several important road sign categories.
        </p>

        <div className="classes-grid">
          <div className="class-card">
            <span className="class-symbol">30</span>
            <h3>Speed Limits</h3>
            <p>
              Signs that indicate maximum permitted vehicle speeds.
            </p>
          </div>

          <div className="class-card">
            <span className="class-symbol">!</span>
            <h3>Warning Signs</h3>
            <p>
              Signs that alert drivers to potential road hazards.
            </p>
          </div>

          <div className="class-card">
            <span className="class-symbol">◇</span>
            <h3>Priority Signs</h3>
            <p>
              Signs that communicate right-of-way and road priority.
            </p>
          </div>

          <div className="class-card">
            <span className="class-symbol">⊘</span>
            <h3>Prohibitory Signs</h3>
            <p>
              Signs that indicate restricted or prohibited actions.
            </p>
          </div>

          <div className="class-card">
            <span className="class-symbol">→</span>
            <h3>Mandatory Signs</h3>
            <p>
              Signs that indicate required directions or movements.
            </p>
          </div>
        </div>
      </section>

      {/* ABOUT */}

      <section className="about-section" id="about">
        <div className="eyebrow">
          BEHIND THE TECHNOLOGY
        </div>

        <h2>Intelligence Behind Every Sign.</h2>

        <p>
          SignSight uses Deep Learning and Computer Vision
          to classify traffic signs from uploaded or captured
          images. The system processes images through a trained
          neural network and displays the predicted sign
          along with its confidence score.
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
        <span>43 sign classes · 30×30 RGB input</span>
        <span>Built with TensorFlow.js · runs client-side</span>
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
