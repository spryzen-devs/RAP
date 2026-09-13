"""
reasoning.py — Part B: the "minimal reasoning layer."

This is a single hand-written decision layer, not a framework and not a
multi-agent system. It has three explicit stages, each a plain Python
function, each independently testable:

    1. classify_intent()   — decide whether /ask needs to call the detector
    2. reason_over_detections() — turn structured detections into a factual,
       rule-based summary (counts, compliance flags) BEFORE any LLM involvement
    3. apply_guardrail()   — decide, in code (not by asking the LLM to judge
       itself), whether there's enough evidence to answer confidently

The LLM (api/llm_client.call_llm) is used for exactly one thing: turning the
already-computed structured summary into a natural-language sentence. It
never decides routing and never decides confidence — those are both
deterministic, auditable, hand-written rules, which is what makes the
guardrail actually reliable rather than just an LLM saying "I'm not sure"
when it feels like it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from api.llm_client import call_llm


def _contains_phrase(text: str, phrase: str) -> bool:
    """Whole-word/whole-phrase substring match using regex word boundaries.

    Plain `phrase in text` is NOT safe here: "image" contains "age" as a
    literal substring, and "construction sites" contains "on site" spanning
    a word break ("constructi[on] [site]s"). \\b anchors at the edges of the
    *whole* phrase reject both of those false positives while still matching
    the intended real occurrences (including multi-word phrases like
    "this image").
    """
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def _any_phrase(text: str, phrases: list[str]) -> bool:
    return any(_contains_phrase(text, p) for p in phrases)

# ---------------------------------------------------------------------------
# Taxonomy knowledge the router/guardrail need. Kept in sync with
# data/dataset.yaml by hand (5 classes — small enough that a generated
# mapping would be overkill and would hide the logic from a verbal defense).
# ---------------------------------------------------------------------------
DETECTOR_CLASSES = ["Hardhat", "NO-Hardhat", "Safety-Vest", "NO-Safety-Vest", "Person"]

# Concepts the detector CAN speak to, and the keywords that signal a question
# is asking about that concept. This is intentionally a small, hand-curated
# lexicon (not an embedding model) — the router needs to be auditable line by
# line for the verbal defense, not a black box.
CONCEPT_KEYWORDS = {
    "hardhat": ["helmet", "hardhat", "hard hat", "headgear"],
    "vest": ["vest", "hi-vis", "hi vis", "high-visibility", "high visibility"],
    "person": ["person", "people", "worker", "workers", "anyone", "someone"],
}

# Attributes people commonly ask about that are visually plausible but that
# our 5-class taxonomy has NO way to answer — these must route to the
# guardrail as "insufficient", not be guessed at by the LLM. Checked with
# HIGHEST priority: even if the same question also names a covered concept
# (e.g. "is the *person* wearing steel-toe *boots*?"), the attribute actually
# being asked about is still out of taxonomy, so this must win over a
# concept-keyword match, not lose to it.
OUT_OF_TAXONOMY_KEYWORDS = [
    "boots", "shoes", "footwear", "gloves", "goggles", "glasses", "mask", "respirator",
    "color", "colour", "brand", "logo", "age", "gender", "male", "female", "name",
    "height", "weight", "mood", "emotion", "expression", "time of day", "weather",
]

# Phrases that mark a question as being about THIS specific uploaded image
# (explicit reference, or a count/existence question that only makes sense
# grounded in a particular image) — these should win over a bare concept
# keyword when deciding whether the question is really general knowledge.
IMAGE_CONTEXT_PHRASES = [
    "this image", "this photo", "this picture", "the image", "the photo", "the picture",
    "in this", "here", "pictured", "shown", "on site", "in the site",
]
COUNT_OR_EXISTENCE_PHRASES = [
    "how many", "is anyone", "is there", "are there", "does the image",
    "what's the most common", "what is the most common", "count the",
]

# Phrasing that marks a question as textbook/general knowledge rather than
# about a specific image — only decisive when NOT combined with an
# image-context or count/existence phrase above.
GENERAL_KNOWLEDGE_PHRASES = [
    "usually", "typically", "in general", "generally", "what is a", "what's a",
    "why do", "why is", "why are", "what does osha", "what is the standard",
]


@dataclass
class IntentDecision:
    needs_detector: bool
    reason: str
    target_concepts: list[str] = field(default_factory=list)
    out_of_taxonomy: bool = False


def classify_intent(question: str) -> IntentDecision:
    """Stage 1 — rule-based router.

    Decision order (each rule is a plain, defensible sentence). Order matters
    and is deliberate — later rules only run if earlier ones didn't already
    decide:

      a) If the question is phrased as general/textbook knowledge (has a
         "usually"/"typically"/"why is" cue) AND doesn't also reference this
         specific image (no "this image"/"how many"/etc.) -> no detector.
         This is checked FIRST so e.g. "What color is a hardhat usually?"
         doesn't get pulled into the detector path just because it names
         "hardhat".
      b) Else if the question asks about a visual attribute outside our
         5-class taxonomy (boots, color, brand, age, ...) -> we still call
         the detector best-effort, but mark out_of_taxonomy=True so the
         guardrail refuses deterministically. Checked BEFORE the
         covered-concept check on purpose: "is the person wearing boots"
         also matches "person", but the thing actually being asked about
         (boots) is what should decide the outcome, not the incidental
         mention of a covered class.
      c) Else if the question references a concept our classes cover
         (hardhat/vest/person) -> call the detector.
      d) Else if the question references this image but names no covered
         concept -> best-effort detector call, flagged out_of_taxonomy so a
         low-evidence answer gets caught by the guardrail.
      e) Otherwise -> general knowledge, not about this image, answer
         directly.
    """
    q = question.lower()

    mentions_image_context = _any_phrase(q, IMAGE_CONTEXT_PHRASES) or _any_phrase(q, COUNT_OR_EXISTENCE_PHRASES)
    is_general_knowledge_phrasing = _any_phrase(q, GENERAL_KNOWLEDGE_PHRASES)

    if is_general_knowledge_phrasing and not mentions_image_context:
        return IntentDecision(
            needs_detector=False,
            reason="phrased as general/textbook knowledge (e.g. 'usually'/'why is') rather than "
                   "a question about this specific image",
        )

    out_of_taxonomy_hit = _any_phrase(q, OUT_OF_TAXONOMY_KEYWORDS)
    matched_concepts = [c for c, kws in CONCEPT_KEYWORDS.items() if _any_phrase(q, kws)]

    if out_of_taxonomy_hit:
        return IntentDecision(
            needs_detector=True,
            reason="question asks about a visual attribute (e.g. footwear/color/age) that is "
                   "outside the detector's Hardhat/Vest/Person taxonomy"
                   + (f" (also mentions covered concept(s): {', '.join(matched_concepts)}, "
                      f"but that isn't what's being asked)" if matched_concepts else ""),
            target_concepts=matched_concepts,
            out_of_taxonomy=True,
        )

    if matched_concepts:
        return IntentDecision(
            needs_detector=True,
            reason=f"question references detector-covered concept(s): {', '.join(matched_concepts)}",
            target_concepts=matched_concepts,
            out_of_taxonomy=False,
        )

    if mentions_image_context:
        return IntentDecision(
            needs_detector=True,
            reason="question refers to 'this image'/'the photo' but doesn't name a "
                   "detector-covered concept — best-effort detector call, likely insufficient",
            target_concepts=[],
            out_of_taxonomy=True,
        )

    return IntentDecision(
        needs_detector=False,
        reason="general knowledge question, not about the content of this specific image",
        target_concepts=[],
        out_of_taxonomy=False,
    )


@dataclass
class StructuredSummary:
    person_count: int
    hardhat_count: int
    no_hardhat_count: int
    vest_count: int
    no_vest_count: int
    mean_confidence: float
    class_counts: dict
    detections_used: int


def reason_over_detections(detections: list[dict]) -> StructuredSummary:
    """Stage 2 — pure, deterministic aggregation. No LLM here. Whatever this
    function returns is exactly what the LLM (stage 3's phrasing step) is
    allowed to talk about — it cannot introduce facts this summary doesn't
    contain, because the prompt only hands it this summary, not raw pixels.
    """
    class_counts: dict[str, int] = {c: 0 for c in DETECTOR_CLASSES}
    confidences = []
    for d in detections:
        class_counts[d["class_name"]] = class_counts.get(d["class_name"], 0) + 1
        confidences.append(d["confidence"])

    return StructuredSummary(
        person_count=class_counts.get("Person", 0),
        hardhat_count=class_counts.get("Hardhat", 0),
        no_hardhat_count=class_counts.get("NO-Hardhat", 0),
        vest_count=class_counts.get("Safety-Vest", 0),
        no_vest_count=class_counts.get("NO-Safety-Vest", 0),
        mean_confidence=round(sum(confidences) / len(confidences), 3) if confidences else 0.0,
        class_counts=class_counts,
        detections_used=len(detections),
    )


MIN_CONFIDENCE_FOR_ANSWER = 0.40  # below this, we don't trust the aggregate enough to answer


@dataclass
class GuardrailResult:
    confidence: str  # "high" | "medium" | "insufficient"
    reason_code: str | None = None


def apply_guardrail(intent: IntentDecision, summary: StructuredSummary | None) -> GuardrailResult:
    """Stage 3a — decide, in code, whether we have enough to answer.

    This runs BEFORE the LLM is asked to phrase anything, and its output
    (confidence) is authoritative — the LLM is never allowed to override it.
    """
    if intent.out_of_taxonomy:
        return GuardrailResult(confidence="insufficient", reason_code="no_matching_class_in_taxonomy")

    if not intent.needs_detector:
        return GuardrailResult(confidence="high")  # general-knowledge answers don't need visual evidence

    if summary is None or summary.detections_used == 0:
        return GuardrailResult(confidence="insufficient", reason_code="zero_detections_returned")

    if summary.mean_confidence < MIN_CONFIDENCE_FOR_ANSWER:
        return GuardrailResult(confidence="insufficient", reason_code="detections_below_confidence_threshold")

    if summary.mean_confidence < 0.6:
        return GuardrailResult(confidence="medium")

    return GuardrailResult(confidence="high")


def answer_question(question: str, detections: list[dict] | None, intent: IntentDecision,
                     guardrail: GuardrailResult, summary: StructuredSummary | None) -> str:
    """Stage 3b — produce the final natural-language sentence.

    - For 'insufficient', we do NOT call the LLM at all — the refusal message
      is generated directly from the reason code, so there is no chance of
      the LLM quietly guessing anyway. This is the actual guardrail
      enforcement point.
    - For 'high'/'medium' with no detector needed, we let the LLM answer a
      general-knowledge question directly.
    - For 'high'/'medium' with detector evidence, the LLM is only shown the
      StructuredSummary (numbers), never the raw image or raw box list, and
      is instructed to restate only what's in that summary.
    """
    if guardrail.confidence == "insufficient":
        reasons = {
            "no_matching_class_in_taxonomy": (
                "I can't answer that confidently — my detector is only trained on "
                f"{', '.join(DETECTOR_CLASSES)} classes and does not detect that attribute, "
                "so there is no evidence in the structured output to support or refute this."
            ),
            "zero_detections_returned": (
                "I can't answer that confidently — the detector did not find anything relevant "
                "in this image at the configured confidence threshold, so I don't have evidence "
                "either way."
            ),
            "detections_below_confidence_threshold": (
                "I can't answer that confidently — the detector produced results, but at a "
                f"mean confidence of {summary.mean_confidence:.2f}, below the "
                f"{MIN_CONFIDENCE_FOR_ANSWER:.2f} threshold this API requires before treating a "
                "detection as reliable enough to base an answer on."
            ),
        }
        return reasons.get(guardrail.reason_code, "I can't answer that confidently given the available evidence.")

    if not intent.needs_detector:
        return call_llm(
            system_prompt=(
                "You are a concise assistant embedded in a construction-site PPE compliance API. "
                "Answer general-knowledge questions directly and factually in 1-2 sentences. "
                "You are not being shown an image for this question."
            ),
            user_prompt=question,
        )

    evidence_text = (
        f"Detected in the image: {summary.person_count} person(s), "
        f"{summary.hardhat_count} Hardhat, {summary.no_hardhat_count} NO-Hardhat, "
        f"{summary.vest_count} Safety-Vest, {summary.no_vest_count} NO-Safety-Vest "
        f"(mean detector confidence {summary.mean_confidence:.2f}, "
        f"{summary.detections_used} raw detections)."
    )
    return call_llm(
        system_prompt=(
            "You are a concise assistant embedded in a construction-site PPE compliance API. "
            "You are given a user question and a factual summary of a computer-vision "
            "detector's output for one image. Answer ONLY using the numbers in that summary — "
            "do not invent details the summary doesn't contain (no colors, no identities, no "
            "attributes beyond hardhat/vest/person counts). Answer in 1-2 plain sentences, "
            "state the relevant counts explicitly."
        ),
        user_prompt=f"Question: {question}\n\n{evidence_text}",
    )
