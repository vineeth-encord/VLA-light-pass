"""
test_vllm.py — smoke-test the vLLM server before running the full agent.

Usage (on the GH200, with the agent venv active):
    python scripts/deploy/test_vllm.py
    python scripts/deploy/test_vllm.py --image path/to/frame.jpg
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import sys

import numpy as np
from PIL import Image, ImageDraw


def _synthetic_frame() -> np.ndarray:
    """Create a simple 640×480 test image when no real frame is provided."""
    img = Image.new("RGB", (640, 480), color=(100, 150, 200))
    draw = ImageDraw.Draw(img)
    draw.rectangle([80, 120, 300, 300], outline="white", width=3)
    draw.text((90, 310), "test manipulation frame", fill="white")
    return np.array(img)


def frame_to_b64(frame: np.ndarray) -> str:
    pil = Image.fromarray(frame)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the vLLM server")
    parser.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--model",    default=os.environ.get("VLLM_MODEL", "qwen-vl"))
    parser.add_argument("--image",    default=None, help="Path to a JPEG/PNG frame (optional)")
    args = parser.parse_args()

    print(f"vLLM base URL : {args.base_url}")
    print(f"Model         : {args.model}")

    # 1. List available models
    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="token-encord")

    print("\n[1] Available models:")
    models = client.models.list()
    for m in models.data:
        print(f"    {m.id}")

    # 2. Load test frame
    if args.image:
        frame = np.array(Image.open(args.image).convert("RGB"))
        print(f"\n[2] Loaded frame from {args.image}: {frame.shape}")
    else:
        frame = _synthetic_frame()
        print(f"\n[2] Using synthetic test frame: {frame.shape}")

    b64 = frame_to_b64(frame)

    # 3. Call the model
    print("\n[3] Sending inference request …")
    response = client.chat.completions.create(
        model=args.model,
        max_tokens=256,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a robot manipulation video annotator. Respond with a single JSON object: "
                    '{"manipulation_phase": "...", "scene_context": "...", "motion_mode": "...", "objects": []}'
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"},
                    },
                    {"type": "text", "text": "Annotate this robot manipulation frame."},
                ],
            },
        ],
    )

    print("\n[4] Model response:")
    print(response.choices[0].message.content)
    print(f"\n    Prompt tokens : {response.usage.prompt_tokens}")
    print(f"    Completion    : {response.usage.completion_tokens}")
    print("\n✅  vLLM smoke-test passed.")


if __name__ == "__main__":
    main()
