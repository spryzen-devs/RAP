# Construction PPE Detection and Reasoning API - Final Memo

## Domain & Dataset Selection
**Domain:** Construction/Industrial Personal Protective Equipment (PPE) Compliance.
**Dataset:** The dataset consists of 2,659 images containing construction workers and equipment. The taxonomy includes 10 classes, satisfying the non-COCO requirement with classes like: `Hardhat`, `Mask`, `NO-Hardhat`, `NO-Mask`, `NO-Safety Vest`, `Safety Cone`, `Safety Vest`, `machinery`. 

**Sourcing & Labeling:** The dataset is a curated mix of scraped construction site imagery and publicly available industrial safety footage. Annotations were provided in YOLO format for bounding boxes covering both compliant items (e.g., `Safety Vest`) and non-compliant states (e.g., `NO-Safety Vest`). 

## Data Split Strategy
The dataset was split using an approximate 70/15/15 ratio:
- **Train:** 1,861 images
- **Validation:** 398 images
- **Test:** 400 images

**Technical Justification:** A standard 70/15/15 split provides sufficient training data for the RT-DETR-L model to generalize across various lighting and occlusion conditions, while reserving a robust validation set for monitoring overfitting during fine-tuning. The 400-image test set is strictly held-out to evaluate final generalizability before assessing against the hidden evaluation set.

## Evaluation Metrics
The model was evaluated using standard Object Detection metrics: **mAP50** and **mAP50-95**. 
- **What they tell us:** mAP50 tells us if the model can accurately find and classify an object with at least a 50% overlap (IoU) with the ground truth. mAP50-95 tells us how precisely the model draws the bounding box. These metrics prove the model learned the visual characteristics of PPE vs non-PPE.
- **What they don't tell us:** These metrics do not reflect the model's operational context or downstream "confusion" risk. For example, a high mAP doesn't reveal if the model relies purely on the color orange to detect a vest (making it fail on an orange shirt), or if it struggles to detect tiny masks on workers in the background.

## Failure Case Root-Cause Analysis
1. **Orange shirts confused for `Safety Vest` (False Positive)**
   - *Root Cause:* The model over-indexes on high-visibility colors (neon orange/yellow) located on a person's torso. It fails to strictly require structural features of a vest (like reflective stripes or seams) when the color signal is overwhelming.
2. **Missing small `Mask` at long distances (False Negative)**
   - *Root Cause:* At long distances, the pixel area of a face mask drops below the minimum receptive field threshold for reliable feature extraction, which is exacerbated when the input image is resized to 640x640 during inference.
3. **Occluded `Hardhat` behind machinery (False Negative)**
   - *Root Cause:* When a worker is standing behind heavy machinery, the top of the head is partially occluded. If the machinery is yellow (like an excavator), the yellow hardhat blends visually into the background, stripping the model of both shape and color cues.
4. **Class confusion between `vehicle` and `machinery`**
   - *Root Cause:* Heavy construction vehicles (e.g., dump trucks) share many visual and textural features with static machinery (treads, hydraulic arms, industrial yellow paint), making the boundary between these two classes inherently ambiguous without motion context.
5. **NMS suppression of `NO-Safety Vest` when standing behind a compliant worker**
   - *Root Cause:* Bounding box overlap. When a worker wearing a vest stands directly in front of a worker without a vest, their bounding boxes heavily overlap. The Non-Maximum Suppression (NMS) algorithm may discard the `NO-Safety Vest` detection, assuming it is a duplicate box for the foreground person.

## Part B: Minimal Reasoning Layer
The reasoning layer (`api/reasoning.py`) is intentionally implemented as a single, hand-written decision layer with strict programmatic routing, avoiding the unpredictability of a multi-agent framework.

**Intent Routing Logic:**
The system uses deterministic regex word-boundary matching (`\b`) to evaluate the user's question:
1. **General Knowledge:** If the question contains cues like "usually" or "why is" (and doesn't reference "this image"), it bypasses the detector and asks the LLM to answer using general knowledge.
2. **Out of Taxonomy:** If the question asks about un-modeled visual attributes (e.g., "boots", "brand", "gender"), it flags the intent as `out_of_taxonomy`.
3. **Visual Query:** If the question references modeled concepts ("hardhat", "vest", "person") or mentions "this image", it calls the RT-DETR model to get structured detections.

**Handling Insufficient Information:**
The confidence guardrail intercepts out-of-taxonomy or low-confidence results *before* the LLM sees them. If the detector lacks the required classes or confidence, the system directly returns a canned refusal instead of allowing the LLM to guess.
- **Example:** If the user asks *"Are the workers wearing steel-toe boots?"*
- **Outcome:** The router detects "boots" as an out-of-taxonomy keyword. The guardrail immediately intercepts the request and outputs: *"I can't answer that confidently — my detector is only trained on Hardhat, NO-Hardhat, Safety-Vest, NO-Safety-Vest, Person classes and does not detect that attribute, so there is no evidence in the structured output to support or refute this."* (The LLM is never invoked).

## API Usage Instructions
To run the project, ensure Docker is installed and run the following command in the root directory:
```bash
docker-compose up --build -d
```
*Note: You must provide a valid `GEMINI_API_KEY` in the `.env` file.*

### Endpoints
**1. POST `/detect`**
Accepts a multipart form data image and returns bounding boxes.
```bash
curl -X POST "http://localhost:8000/detect" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"
```
**Response:**
```json
{"detections": [{"box": [100, 150, 200, 300], "confidence": 0.85, "class_id": 0, "class_name": "Hardhat"}]}
```

**2. POST `/ask`**
Accepts a multipart form data image and a natural language question.
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg" \
  -F "question=How many people are not wearing safety vests?"
```
**Response:**
```json
{"answer": "Detected in the image are 2 people, and 1 of them is marked as NO-Safety-Vest."}
```
