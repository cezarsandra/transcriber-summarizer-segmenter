"""
LLM client utilities.

Supports two backends:
  - gemini  : Google Gemini API
  - runpod  : RunPod serverless worker (custom handler format)
"""

import time
import json
import requests


DEFAULT_SAMPLING_PARAMS = {
    "max_tokens": 16000,
    "temperature": 0.1,
    "top_p": 0.9,
}


def _runpod_base_url(url: str) -> str:
    """Strip trailing /run or /runsync to get the base endpoint URL."""
    for suffix in ("/runsync", "/run"):
        if url.rstrip("/").endswith(suffix):
            return url.rstrip("/")[: -len(suffix)]
    return url.rstrip("/")


def call_runpod(
    messages: list,
    endpoint_url: str,
    api_key: str,
    sampling_params: dict | None = None,
    max_tokens: int | None = None,
) -> str:
    """
    Submit a job to a RunPod serverless LLM worker and return the response text.

    endpoint_url: full URL to the worker, e.g.
        https://api.runpod.ai/v2/<endpoint-id>/run
        https://api.runpod.ai/v2/<endpoint-id>          (base, /run appended)
    """
    base = _runpod_base_url(endpoint_url)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    params = dict(sampling_params or DEFAULT_SAMPLING_PARAMS)
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    payload = {
        "input": {
            "messages": messages,
            "sampling_params": params,
        }
    }

    # Submit job
    resp = requests.post(f"{base}/run", json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    job_id = resp.json()["id"]
    print(f"    RunPod job submitted: {job_id}")

    # Poll until done
    while True:
        status_resp = requests.get(f"{base}/status/{job_id}", headers=headers, timeout=30)
        status_resp.raise_for_status()
        status = status_resp.json()

        if status["status"] == "COMPLETED":
            return _extract_text(status.get("output", ""))
        if status["status"] in ("FAILED", "CANCELLED"):
            raise RuntimeError(f"RunPod job {job_id} ended with status: {status['status']}")

        time.sleep(3)


def clean_json_response(text: str) -> str:
    """Strip markdown code fences and leading/trailing noise from an LLM JSON response."""
    import re
    # Remove ```json ... ``` or ``` ... ```
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.replace("```", "")
    # Find first [ or { and last ] or }
    first = min(
        (text.find(c) for c in ("[", "{") if text.find(c) != -1),
        default=-1,
    )
    last = max(text.rfind("]"), text.rfind("}"))
    if first != -1 and last != -1 and last > first:
        text = text[first : last + 1]
    return text.strip()


def _extract_text(output) -> str:
    """Extract response text from various RunPod output shapes."""
    if isinstance(output, str):
        return output

    if isinstance(output, dict):
        choices = output.get("choices")
        if choices and isinstance(choices, list):
            choice = choices[0]
            # vLLM token list: {"choices": [{"tokens": ["...", "...", ...]}]}
            tokens = choice.get("tokens")
            if tokens and isinstance(tokens, list):
                return "".join(tokens)
            # OpenAI-style: {"choices": [{"message": {"content": "..."}}]}
            content = choice.get("message", {}).get("content")
            if content:
                return content
            # Plain text field
            return choice.get("text", "") or choice.get("content", "")
        if "text" in output:
            return output["text"]
        if "content" in output:
            return output["content"]

    if isinstance(output, list) and output:
        first = output[0]
        if isinstance(first, dict):
            return first.get("text", "") or first.get("content", "") or str(first)

    return str(output)


def call_gemini(
    prompt: str,
    system_instruction: str,
    api_key: str,
    model: str,
) -> str:
    """Call Gemini and return response text. Raises on empty response."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
        ),
    )
    if not response.text:
        finish = (
            getattr(response.candidates[0], "finish_reason", "unknown")
            if response.candidates
            else "no candidates"
        )
        raise RuntimeError(f"Empty response from Gemini (finish_reason={finish})")
    return response.text
