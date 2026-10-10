"""Make a Keras 3 TensorFlow.js export readable by the browser LayersModel loader."""

from pathlib import Path


model_path = Path("public/model/model.json")
content = model_path.read_text(encoding="utf-8")
content = content.replace('"batch_shape"', '"batchInputShape"')
content = content.replace("sequential/", "")
model_path.write_text(content, encoding="utf-8")
print(f"Normalized {model_path}")
