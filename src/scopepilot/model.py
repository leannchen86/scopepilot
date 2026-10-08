"""Ways to put a question and some images to a Claude model and get JSON back.

Two backends share one small interface so the rest of the code does not care
which is in use:

- `AnthropicBackend` calls the Claude API and needs API credentials.
- `ClaudeCodeBackend` runs the Claude Code command-line tool, so it works with
  a Claude login and no API key. It is slower and gives less control over the
  image the model sees.
"""

from __future__ import annotations

import base64
import io
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

DEFAULT_MODEL = "claude-opus-5-5"


class ModelError(RuntimeError):
    """The model did not return a usable answer."""


class Backend(Protocol):
    def ask(
        self, system: str, prompt: str, images: Sequence[Image.Image], schema: dict[str, Any]
    ) -> dict[str, Any]:
        """Send the images and prompt; return JSON matching `schema`."""
        ...


def _png_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return base64.standard_b64encode(buffer.getvalue()).decode("ascii")


class AnthropicBackend:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        effort: str = "medium",
        fallbacks: bool = True,
        client: Any = None,
    ) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self._client = client
        self.model = model
        self.effort = effort
        # A safety classifier can decline a harmless request. With fallbacks on,
        # the API retries a declined request on another model inside the same call.
        self.fallbacks = fallbacks

    def ask(
        self, system: str, prompt: str, images: Sequence[Image.Image], schema: dict[str, Any]
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": _png_base64(image)},
            }
            for image in images
        ]
        content.append({"type": "text", "text": prompt})
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 16000,
            "system": system,
            "messages": [{"role": "user", "content": content}],
            "output_config": {
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": schema},
            },
        }
        if self.fallbacks:
            response = self._client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **request
            )
        else:
            response = self._client.messages.create(**request)

        if response.stop_reason == "refusal":
            raise ModelError("the model declined the request")
        if response.stop_reason == "max_tokens":
            raise ModelError("the answer was cut off before it finished")
        text = next((block.text for block in response.content if block.type == "text"), None)
        if text is None:
            raise ModelError("the response contained no text")
        return json.loads(text)


class ClaudeCodeBackend:
    def __init__(
        self, model: str = DEFAULT_MODEL, executable: str = "claude", timeout: float = 300
    ) -> None:
        self.model = model
        self.executable = executable
        self.timeout = timeout

    def ask(
        self, system: str, prompt: str, images: Sequence[Image.Image], schema: dict[str, Any]
    ) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="scopepilot-") as tmp:
            lines = []
            for number, image in enumerate(images, start=1):
                path = Path(tmp) / f"image-{number}.png"
                image.convert("RGB").save(path)
                lines.append(f"Image {number}: {path}")
            listing = "\n".join(lines)
            command = [
                self.executable,
                "--print",
                f"Open each of these image files with the Read tool before answering.\n{listing}\n\n{prompt}",
                "--output-format", "json",
                "--json-schema", json.dumps(schema),
                "--system-prompt", system,
                "--tools", "Read",
                "--allowedTools", "Read",
                "--add-dir", tmp,
                "--model", self.model,
                "--no-session-persistence",
            ]  # fmt: skip
            try:
                done = subprocess.run(
                    command, capture_output=True, text=True, timeout=self.timeout, cwd=tmp
                )
            except subprocess.TimeoutExpired as error:
                raise ModelError(f"claude did not answer within {self.timeout:.0f}s") from error
        if done.returncode != 0 and not done.stdout.strip():
            raise ModelError(f"claude exited with {done.returncode}: {done.stderr.strip()[:500]}")
        # On failure claude still prints its JSON envelope, with the reason in it.
        return _parse_claude_code_output(done.stdout)


def _parse_claude_code_output(stdout: str) -> dict[str, Any]:
    try:
        envelope = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise ModelError(f"claude printed something that is not JSON: {stdout[:300]}") from error
    if envelope.get("is_error"):
        raise ModelError(f"claude reported an error: {str(envelope.get('result'))[:500]}")
    structured = envelope.get("structured_output")
    if isinstance(structured, dict):
        return structured
    try:
        return json.loads(envelope["result"])
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ModelError("claude returned no structured answer") from error


def default_backend(model: str = DEFAULT_MODEL) -> Backend:
    """API credentials if the environment has them, otherwise the Claude Code login."""
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return AnthropicBackend(model=model)
    if shutil.which("claude"):
        return ClaudeCodeBackend(model=model)
    raise ModelError(
        "no way to reach a model: set ANTHROPIC_API_KEY, or install Claude Code and log in"
    )
