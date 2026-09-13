<div align="center">
  <img src="https://img.icons8.com/color/96/000000/worker-male.png" alt="Worker"/>
  <h1>🛡️ AegisSight </h1>
  <h3>Intelligent Construction Safety Guardian</h3>
  <p><i>Next-Generation PPE Vision AI powered by RT-DETR & Natural Language Reasoning</i></p>

  <p>
    <a href="#features">Features</a> •
    <a href="#quick-start">Quick Start</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#api-usage">API Usage</a> •
    <a href="#model-training">Model Training</a>
  </p>
</div>

---

Welcome to **AegisSight**! A cutting-edge AI system designed to ensure the safety of construction sites. By analyzing site images in real-time, AegisSight autonomously verifies if workers are equipped with their mandatory Personal Protective Equipment (PPE) such as hardhats and safety vests.

Beyond simple bounding boxes, AegisSight features a **Natural Language Reasoning Layer**. You can interact with the system in plain English (e.g., *"Is anyone missing a helmet?"*), and it will provide clear, evidence-based answers.

---

## ✨ Features

- 🎯 **Real-Time Detection**: Powered by a fine-tuned **RT-DETR-L** (Real-Time DEtection TRansformer) model for high-precision, low-latency object detection.
- 💬 **Natural Language Querying**: Ask questions about the scene in plain English, powered by a deterministic intent router + Gemini LLM.
- 🛡️ **Fail-Safe Reasoning**: The system gracefully handles out-of-taxonomy queries or low-confidence detections by explicitly refusing to guess.
- ⚡ **FastAPI Backend**: A lightning-fast, production-ready REST API.
- 🐳 **Dockerized**: Fully containerized for one-command deployments.
- 📊 **Beautiful Web UI**: Includes an interactive web dashboard for real-time inference and Q&A.

---

## 🏗️ What Can It See?

AegisSight is trained on a curated dataset of over 2,650 construction site images to identify compliance and violations across key safety categories:

- 👷 **Hardhat** & 🚫 **NO-Hardhat**
- 🦺 **Safety-Vest** & 🚫 **NO-Safety-Vest**
- 😷 **Mask** & 🚫 **NO-Mask**
- 🧍 **Person**
- 🚧 **Safety Cone** & 🚜 **Machinery**

By modeling both the *presence* and *absence* of gear, the AI excels in real-world safety monitoring.

---

## 🚀 Quick Start

Get up and running in minutes using Docker.

### 1. Set Environment Variables
```bash
cp .env.example .env
# Open .env and add your GEMINI_API_KEY for the reasoning engine
```

### 2. Launch with Docker Compose
```bash
# Start the backend API and the web dashboard
docker-compose up --build
```
> [!TIP]
> Once the container is running, navigate to `http://localhost:8000` in your browser to access the interactive web dashboard!

---

## 🧠 System Architecture

AegisSight is designed to bridge the gap between deterministic object detection and flexible natural language processing:

1. **Vision Layer (RT-DETR)**: The image is processed by the RT-DETR model to extract structured data (bounding boxes, classes, confidence scores).
2. **Intent Router**: A regex-based reasoning layer analyzes the user's natural language query to determine if it's a visual query, general knowledge, or out-of-taxonomy.
3. **Guardrails**: If the user asks about an object the model wasn't trained on (e.g., "boots"), the system intelligently intercepts and refuses to hallucinate.
4. **Synthesis (LLM)**: Valid visual queries are sent to the LLM along with the structured detection data to formulate a human-friendly response.

---

## 💻 API Usage

Interact programmatically with AegisSight via the REST API.

### `POST /detect`
Upload an image to get raw bounding boxes and confidence scores.
```bash
curl -X POST "http://localhost:8000/detect" \
     -H "accept: application/json" \
     -F "file=@tests/sample_requests/site_photo_01.jpg"
```

### `POST /ask`
Upload an image and ask a question in natural language.
```bash
curl -X POST "http://localhost:8000/ask" \
     -H "accept: application/json" \
     -F "file=@tests/sample_requests/site_photo_01.jpg" \
     -F "question=How many people are not wearing safety vests?"
```
**Response:**
```json
{
  "answer": "Detected in the image are 2 people, and 1 of them is marked as NO-Safety-Vest."
}
```

---

## 🔬 Model Training & Evaluation

Want to dive into the data science? Here's how AegisSight was built:

- **Dataset**: 2,659 curated images, split into Train (1,861), Validation (398), and Test (400).
- **Architecture**: RT-DETR-L (Large variant) fine-tuned for 80 epochs at 640x640 resolution.
- **Evaluation**: Evaluated on strict mAP50 and mAP50-95 metrics. We conducted extensive root-cause analysis on failure cases (e.g., occlusion by heavy machinery, false positives from orange shirts) to understand the model's limitations beyond aggregate numbers.

### Train it Yourself
```bash
# 1. Download dataset via Roboflow
python data/prepare_dataset.py --roboflow-key $ROBOFLOW_API_KEY --out data/processed

# 2. Train the model
python src/train.py --data data/dataset.yaml --model rtdetr-l.pt --epochs 80

# 3. Evaluate performance
python src/evaluate.py --weights runs/ppe/best.pt --data data/dataset.yaml
```

---

<div align="center">
  <i>Stay safe out there. 🏗️ Built with passion for a safer tomorrow.</i>
</div>
