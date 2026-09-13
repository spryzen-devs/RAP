# Five Failure Cases — Root Cause Analysis

> **Status: TEMPLATE.** I (the assistant) cannot train the real model in this
> sandbox (no GPU, no internet), so these are not real crops from a real run —
> they are the five failure *categories* you are almost guaranteed to hit
> with this dataset/architecture combination, each with the specific thing to
> look for, how to diagnose it, and what evidence to capture. Run
> `src/evaluate.py`, look at `runs/ppe/<name>/confusion_matrix.png` plus the
> false-positive/false-negative crops it saves, pick one real example per
> category below, replace the `[[FILL IN]]` blocks with the actual image
> crop + numbers, and delete this warning line before submitting.
>
> Do not submit this file with `[[FILL IN]]` still in it — an unfilled
> template is worse than three good real cases, per the "zero acknowledged
> failure cases is a red flag" grading note; a fabricated one is worse still.

---

## 1. Small / distant hardhats missed (false negative)

**Symptom:** Person detected confidently, but a hardhat that occupies only a
few dozen pixels (worker in the background, far from camera) is not detected
at all — the person is scored as if unhelmeted even when they aren't.

**Root cause:** RT-DETR, like most transformer detectors, uses a fixed-size
set of object queries and a feature pyramid whose finest stride still
subsamples small objects heavily; a helmet that's ~20x20px at 640px input
resolution has very little signal left by the deepest feature map it's
matched against. The base dataset also skews toward medium/close-up subjects
(see `data/README.md`), so the model has seen comparatively few small-hardhat
positives during fine-tuning.

**Evidence:** `[[FILL IN: crop + confidence score + ground truth box]]`

**What would fix it:** increase input resolution (960 instead of 640) for
inference on wide-shot images specifically, or tile large images and run
detection per-tile — at the cost of latency, which matters for the API's
`inference_ms` field, so this is a genuine trade-off to note in the memo
rather than a free win.

---

## 2. NO-Hardhat / Hardhat confusion on partial occlusion

**Symptom:** A worker with a hardhat that's partially occluded by scaffolding,
another worker, or their own raised arm gets classified `NO-Hardhat` with
moderate confidence, or gets two overlapping boxes (`Hardhat` and
`NO-Hardhat`) on the same person.

**Root cause:** The two classes are visually adjacent by design (that's the
whole point of the taxonomy), so any occlusion that hides the diagnostic
region (the dome/brim of the helmet) pushes the decision toward whichever
class the visible skin/hair/shadow pattern more closely resembles in the
training distribution. This is a genuinely hard case, not a bug — it's the
system correctly having low information, just not correctly *reporting* low
information at the per-box level (see gap noted in `docs/memo.docx` §5 about
per-box vs. aggregate confidence).

**Evidence:** `[[FILL IN: crop + both candidate boxes + confidences]]`

**What would fix it:** class-aware NMS that suppresses low-confidence
Hardtat/NO-Hardhat duplicates on the same person rather than treating them as
independent detections; more occlusion-heavy training crops (deliberately
crop training images to simulate partial occlusion as an augmentation).

---

## 3. Motion blur on moving workers/machinery

**Symptom:** Frames pulled from short video clips (see grouping note in
`data/README.md`) where a worker is mid-stride show a marked drop in
detection confidence across all classes, sometimes below the API's
`DETECT_CONF_THRESHOLD` entirely — a real hardhat violation goes unreported
not because the model doesn't "know" what a hardhat looks like, but because
the input signal itself is degraded.

**Root cause:** RT-DETR's backbone is trained on (and fine-tuned from) mostly
sharp COCO-style imagery; motion blur shifts the input distribution in a way
fine-tuning on a still-image-heavy dataset doesn't correct for.

**Evidence:** `[[FILL IN: crop + confidence vs. a sharp-frame comparison]]`

**What would fix it:** blur-augmentation during training (synthetic motion
blur on a subset of training images) rather than relying on the source
dataset to contain enough real blurred examples naturally.

---

## 4. Lighting extremes (backlighting / harsh midday shadow)

**Symptom:** Precision drops specifically on images shot with the sun behind
the subject, or with hard shadow across the upper body — vest color (the
main visual cue for Safety-Vest / NO-Safety-Vest) becomes ambiguous in deep
shadow, producing false `NO-Safety-Vest` calls on workers who are actually
compliant.

**Root cause:** High-visibility vest classification leans heavily on
color/contrast cues that degrade non-linearly under extreme dynamic range;
the base dataset (per `data/README.md`) is mostly clean daylight shots, so
the model hasn't been exposed to much of this during fine-tuning — this is
exactly the gap the supplementary scraped images were meant to partially
close, and this failure case is the honest measure of how much they actually
helped.

**Evidence:** `[[FILL IN: crop + predicted class + true class]]`

**What would fix it:** targeted collection of backlit/harsh-shadow site
photos (not just more data in general — more data *of this specific
condition*), and/or HSV-space color-jitter augmentation tuned to simulate
under/over-exposure rather than generic brightness jitter.

---

## 5. Class confusion: `machinery`/background clutter misclassified as `Person`

**Symptom:** On busy sites with scaffolding, stacked materials, or parked
equipment, the model occasionally raises a low-to-moderate confidence
`Person` box on a non-person silhouette (mannequin-shaped stack of materials,
a machinery operator's cab reflection, etc.).

**Root cause:** `Person` is the class with by far the most training examples
and the widest pose/appearance variety in the base dataset, so the decision
boundary for "Person" is comparatively loose compared to the narrower PPE
classes; visually person-shaped clutter on a busy site is exactly the kind
of hard negative that's under-represented if the dataset is mostly
close-up/posed worker photos rather than genuinely cluttered wide shots.

**Evidence:** `[[FILL IN: crop + confidence + what it actually is]]`

**What would fix it:** explicitly mine hard-negative background clutter from
the source dataset's wide-shot images (even ones with zero PPE-relevant
annotations) and include them as negative examples during training, rather
than only including images that already have a target-class box in them.

---

## What this means for the reasoning layer (Part B)

Cases 1, 3, and 4 all manifest as *low mean confidence* rather than *zero
detections* — which is exactly why `api/reasoning.py`'s guardrail checks
`mean_confidence < MIN_CONFIDENCE_FOR_ANSWER` rather than only checking
`detections_used == 0`. A question like "is anyone not wearing a hardhat?"
asked against a blurry or backlit frame should come back `insufficient`, not
a confident wrong answer — that's the direct link between this failure
analysis and the guardrail design, and is a good thing to walk through in
the verbal defense.
