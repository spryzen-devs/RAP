"""
llm_client.py — A deliberately thin, hand-written wrapper around a single LLM
HTTP call. This is NOT an agent framework: there is no tool-calling loop, no
planner, no memory, no multi-step orchestration here. It does exactly one
thing — send a prompt, get text back — and every decision about *when* to
call it and *what* to do with the result lives in reasoning.py, written by
hand. This separation is deliberate so the "hard constraint: no agentic
frameworks" is satisfiable while still using an LLM for the natural-language
part of Part B, per the problem statement's explicit allowance of AI tools.
"""
from __future__ import annotations

import os

import httpx

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

def call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 300, temperature: float = 0.0) -> str:
    """Single blocking call to the Gemini API. Returns plain text."""
    api_key = os.environ.get("GEMINI_API_KEY")
    model = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in your .env file.")

    url = GEMINI_API_URL.format(model=model)
    
    response = httpx.post(
        url,
        params={"key": api_key},
        headers={"Content-Type": "application/json"},
        json={
            "systemInstruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents": [{
                "parts": [{"text": user_prompt}]
            }],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            }
        },
        timeout=30.0,
    )
    response.raise_for_status()
    data = response.json()
    return data["candidates"][0]["content"]["parts"][0]["text"].strip()
