from __future__ import annotations

import json
import os
import re

from openai import OpenAI


def _get_client() -> OpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY environment variable not set")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def create_sentences(source: str, target: str) -> tuple[list[str], list[str]]:
    client = _get_client()

    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert assistant for Pix2Pix-Zero zero-shot image-to-image translation. "
                    "Output ONLY a valid JSON array of objects, each with keys \"source\" and \"target\". "
                    "No extra text or markdown."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Generate 60 diverse source-target sentence pairs for the semantic edit "
                    f"from \"{source}\" to \"{target}\". "
                    f"Each pair must have identical grammatical structure — the ONLY difference "
                    f"between the source and target sentence must be the concept swap "
                    f"(\"{source}\" → \"{target}\"). "
                    f"Cover diverse scenarios: different locations (indoor, outdoor, urban, nature), "
                    f"actions (sitting, standing, running, sleeping, playing, eating, looking), "
                    f"angles (close-up, portrait, from above, from the side, wide shot), "
                    f"lighting (sunlight, shade, night, daylight, golden hour, overcast), "
                    f"weather (rain, snow, fog), compositions (centered, left, right, blurred background), "
                    f"contexts (with people, near objects, on furniture, in specific rooms). "
                    f"Each sentence must start with \"a photo of\" or \"a photograph of\". "
                    f"Return ONLY the JSON array."
                ),
            },
        ],
        stream=False,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}},
    )

    content = response.choices[0].message.content.strip()

    try:
        pairs = json.loads(content)
    except json.JSONDecodeError:
        matches = re.findall(
            r'\{[^}]*"source"\s*:\s*"([^"]*)"\s*,\s*"target"\s*:\s*"([^"]*)"[^}]*\}',
            content,
        )
        if matches:
            pairs = [{"source": s, "target": t} for s, t in matches]
        else:
            raise RuntimeError(f"Could not parse API response: {content[:300]}")

    if not isinstance(pairs, list) or not all(
        isinstance(p, dict) and "source" in p and "target" in p for p in pairs
    ):
        raise RuntimeError(f"Unexpected API response format: {content[:200]}")

    source_sentences: list[str] = []
    target_sentences: list[str] = []
    seen: set[tuple[str, str]] = set()
    for p in pairs:
        key = (p["source"], p["target"])
        if key not in seen:
            seen.add(key)
            source_sentences.append(p["source"])
            target_sentences.append(p["target"])

    return source_sentences, target_sentences
