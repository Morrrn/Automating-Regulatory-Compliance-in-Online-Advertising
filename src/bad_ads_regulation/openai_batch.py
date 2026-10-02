from __future__ import annotations

import json
import os
import uuid
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _api_key(api_key: str | None = None) -> str:
    # read the api key at runtime
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return key


def _request_json(url: str, *, method: str, api_key: str | None = None, body: dict[str, Any] | None = None, timeout: int = 120) -> dict[str, Any]:
    # encode the optional body and prepare an authenticated request
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {_api_key(api_key)}",
            "Content-Type": "application/json",
        },
        method=method,
    )
    # return decoded json and expose useful http error details
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API HTTP {exc.code}: {err_body}") from exc


def upload_batch_file(path: str | Path, api_key: str | None = None, timeout: int = 300) -> dict[str, Any]:
    # build a multipart upload with the required batch purpose
    path = Path(path)
    boundary = f"----badads{uuid.uuid4().hex}"
    file_bytes = path.read_bytes()
    parts = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n".encode("utf-8"),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\nContent-Type: application/jsonl\r\n\r\n".encode("utf-8"),
        file_bytes,
        f"\r\n--{boundary}--\r\n".encode("utf-8"),
    ]
    data = b"".join(parts)
    # send the jsonl file to the files api
    req = urllib.request.Request(
        "https://api.openai.com/v1/files",
        data=data,
        headers={
            "Authorization": f"Bearer {_api_key(api_key)}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(data)),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API HTTP {exc.code}: {err_body}") from exc


def create_batch(input_file_id: str, endpoint: str = "/v1/responses", api_key: str | None = None, metadata: dict[str, str] | None = None) -> dict[str, Any]:
    # create an asynchronous batch with a twenty-four-hour window
    body: dict[str, Any] = {
        "input_file_id": input_file_id,
        "endpoint": endpoint,
        "completion_window": "24h",
    }
    if metadata:
        body["metadata"] = metadata
    return _request_json("https://api.openai.com/v1/batches", method="POST", api_key=api_key, body=body)


def retrieve_batch(batch_id: str, api_key: str | None = None) -> dict[str, Any]:
    # fetch the current batch job details
    return _request_json(f"https://api.openai.com/v1/batches/{batch_id}", method="GET", api_key=api_key)


def download_file_content(file_id: str, out_path: str | Path, api_key: str | None = None, timeout: int = 300) -> Path:
    # prepare the output path and authenticated download request
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        f"https://api.openai.com/v1/files/{file_id}/content",
        headers={"Authorization": f"Bearer {_api_key(api_key)}"},
        method="GET",
    )
    # save the raw response body and expose useful http errors
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out_path.write_bytes(resp.read())
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API HTTP {exc.code}: {err_body}") from exc
    return out_path
