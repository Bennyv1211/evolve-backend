import os
import io
import uuid
import base64
import asyncio
import json
import logging
import mimetypes
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Literal
from collections import defaultdict, deque
from urllib.parse import urlencode

import requests
import firebase_admin
from PIL import Image
from fastapi import FastAPI, APIRouter, HTTPException, Depends, status, UploadFile, File, Form, Request
from fastapi.responses import Response, RedirectResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from firebase_admin import auth as firebase_auth_admin, credentials, firestore
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

# Config
FIREBASE_SERVICE_ACCOUNT_JSON = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON", "").strip()
FIREBASE_PROJECT_ID = os.environ.get("FIREBASE_PROJECT_ID", "").strip()
DAILY_PROMPT_LIMIT = int(os.environ.get("DAILY_PROMPT_LIMIT", "10"))
FREE_PROMPT_LIMIT = int(os.environ.get("FREE_PROMPT_LIMIT", "3"))
OPENAI_TEXT_MODEL = os.environ.get("OPENAI_TEXT_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_API_BASE_URL = os.environ.get("OPENAI_API_BASE_URL", "https://api.openai.com/v1").rstrip("/")
KIE_IMAGE_MODEL = os.environ.get("KIE_IMAGE_MODEL", "gpt-image-2")
KIE_MAX_IMAGES_PER_REQUEST = int(os.environ.get("KIE_MAX_IMAGES_PER_REQUEST", "1"))
KIE_API_KEY = os.environ.get("KIE_API_KEY", "")
KIE_API_BASE_URL = os.environ.get("KIE_API_BASE_URL", "https://api.kie.ai").rstrip("/")
KIE_UPLOAD_BASE_URL = os.environ.get("KIE_UPLOAD_BASE_URL", "https://kieai.redpandaai.co").rstrip("/")
KIE_IMAGE_RESOLUTION = os.environ.get("KIE_IMAGE_RESOLUTION", "1K")
KIE_IMAGE_FORMAT = os.environ.get("KIE_IMAGE_FORMAT", "png")
KIE_IMAGE_ASPECT_RATIO = os.environ.get("KIE_IMAGE_ASPECT_RATIO", "auto")
KIE_POLL_INTERVAL_SECONDS = float(os.environ.get("KIE_POLL_INTERVAL_SECONDS", "2.5"))
KIE_POLL_TIMEOUT_SECONDS = int(os.environ.get("KIE_POLL_TIMEOUT_SECONDS", "180"))
META_APP_ID = os.environ.get("META_APP_ID", "").strip()
META_APP_SECRET = os.environ.get("META_APP_SECRET", "").strip()
META_REDIRECT_URI = os.environ.get("META_REDIRECT_URI", "").strip()
META_GRAPH_API_BASE_URL = os.environ.get("META_GRAPH_API_BASE_URL", "https://graph.facebook.com/v23.0").rstrip("/")
META_OAUTH_BASE_URL = os.environ.get("META_OAUTH_BASE_URL", "https://www.facebook.com/v23.0").rstrip("/")
META_APP_REDIRECT_URI = os.environ.get("META_APP_REDIRECT_URI", "evolve://connect").strip()

# --- Abuse / cost protection knobs ---
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))  # 5 MB
MAX_IMAGE_DIMENSION = int(os.environ.get("MAX_IMAGE_DIMENSION", "1536"))           # px on longest side
MAX_PROMPT_CHARS = int(os.environ.get("MAX_PROMPT_CHARS", "1000"))
MAX_AI_PER_MINUTE_PER_USER = int(os.environ.get("MAX_AI_PER_MINUTE_PER_USER", "3"))
GLOBAL_DAILY_AI_CAP = int(os.environ.get("GLOBAL_DAILY_AI_CAP", "500"))            # all users combined
REGISTRATIONS_PER_HOUR_PER_IP = int(os.environ.get("REGISTRATIONS_PER_HOUR_PER_IP", "5"))

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("evolve")

_firestore_client = None


def _initialize_firebase():
    global _firestore_client
    if _firestore_client is not None:
        return _firestore_client
    if not FIREBASE_SERVICE_ACCOUNT_JSON:
        logger.warning("Firebase Admin not configured: FIREBASE_SERVICE_ACCOUNT_JSON missing")
        return None
    try:
        if not firebase_admin._apps:
            options = {"projectId": FIREBASE_PROJECT_ID} if FIREBASE_PROJECT_ID else None
            if FIREBASE_SERVICE_ACCOUNT_JSON.startswith("{"):
                cred = credentials.Certificate(json.loads(FIREBASE_SERVICE_ACCOUNT_JSON))
            else:
                cred = credentials.Certificate(FIREBASE_SERVICE_ACCOUNT_JSON)
            firebase_admin.initialize_app(cred, options)
        _firestore_client = firestore.client()
        return _firestore_client
    except Exception:
        logger.exception("Failed to initialize Firebase Admin")
        return None


def _firestore_db():
    return _initialize_firebase()


def _fs_doc_id(*parts: str) -> str:
    return ":".join(parts)


async def fs_get(collection: str, doc_id: str) -> Optional[dict]:
    client = _firestore_db()
    if client is None:
        raise HTTPException(status_code=503, detail="Firebase Admin is not configured on the server yet.")

    def _read():
        snap = client.collection(collection).document(doc_id).get()
        if not snap.exists:
            return None
        data = snap.to_dict() or {}
        data["id"] = doc_id
        return data

    return await asyncio.to_thread(_read)


async def fs_set(collection: str, doc_id: str, payload: dict, *, merge: bool = True):
    client = _firestore_db()
    if client is None:
        raise HTTPException(status_code=503, detail="Firebase Admin is not configured on the server yet.")

    def _write():
        client.collection(collection).document(doc_id).set(payload, merge=merge)

    await asyncio.to_thread(_write)


async def fs_delete(collection: str, doc_id: str):
    client = _firestore_db()
    if client is None:
        raise HTTPException(status_code=503, detail="Firebase Admin is not configured on the server yet.")

    def _delete():
        client.collection(collection).document(doc_id).delete()

    await asyncio.to_thread(_delete)


async def fs_query(
    collection: str,
    *,
    filters: Optional[List[tuple[str, str, object]]] = None,
    order_by: Optional[str] = None,
    descending: bool = False,
    limit: Optional[int] = None,
) -> List[dict]:
    client = _firestore_db()
    if client is None:
        raise HTTPException(status_code=503, detail="Firebase Admin is not configured on the server yet.")

    def _run():
        query = client.collection(collection)
        for field, op, value in filters or []:
            query = query.where(field, op, value)
        if order_by:
            direction = firestore.Query.DESCENDING if descending else firestore.Query.ASCENDING
            query = query.order_by(order_by, direction=direction)
        if limit:
            query = query.limit(limit)
        docs = []
        for snap in query.stream():
            data = snap.to_dict() or {}
            data["id"] = data.get("id") or snap.id
            docs.append(data)
        return docs

    return await asyncio.to_thread(_run)


async def fs_user_upsert_from_firebase(firebase_uid: str, email: str, name: str) -> dict:
    existing = await fs_get("users", firebase_uid)
    now = utcnow().isoformat()
    doc = {
        "id": firebase_uid,
        "firebase_uid": firebase_uid,
        "email": email,
        "name": name,
        "onboarded": existing.get("onboarded", False) if existing else False,
        "free_prompts_used": int(existing.get("free_prompts_used", 0)) if existing else 0,
        "subscription_status": existing.get("subscription_status", "free") if existing else "free",
        "subscription_expires_at": existing.get("subscription_expires_at") if existing else None,
        "created_at": existing.get("created_at", now) if existing else now,
        "last_login_at": now,
    }
    await fs_set("users", firebase_uid, doc, merge=True)
    return doc

# App
app = FastAPI(title="Evolve Content Creator API")
api_router = APIRouter(prefix="/api")
security = HTTPBearer(auto_error=False)


# ---------- Helpers ----------
def utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def get_current_user(creds: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> dict:
    if not creds or not creds.credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    if _initialize_firebase() is None:
        raise HTTPException(status_code=503, detail="Firebase Admin is not configured on the server yet.")
    try:
        decoded = await asyncio.to_thread(firebase_auth_admin.verify_id_token, creds.credentials)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired Firebase token")
    user_id = decoded.get("uid")
    email = (decoded.get("email") or "").strip().lower()
    if not user_id or not email:
        raise HTTPException(status_code=401, detail="Firebase token is missing uid/email")
    user = await fs_get("users", user_id)
    if not user:
        display_name = (decoded.get("name") or email.split("@")[0]).strip()[:80]
        user = await fs_user_upsert_from_firebase(user_id, email, display_name)
    return user


def today_key() -> str:
    return utcnow().strftime("%Y-%m-%d")


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def is_user_subscribed(user: dict) -> bool:
    status = str(user.get("subscription_status") or "free").strip().lower()
    if status not in {"active", "trialing", "grace_period"}:
        return False
    expires = _parse_datetime(user.get("subscription_expires_at"))
    if expires is None:
        return True
    return expires > utcnow()


async def verify_subscription_receipt(*_args, **_kwargs) -> bool:
    """Placeholder until Play receipt validation is wired."""
    return False


async def update_subscription_status(
    user_id: str,
    subscription_status: str,
    subscription_expires_at: Optional[str] = None,
):
    await fs_set(
        "users",
        user_id,
        {
            "subscription_status": subscription_status,
            "subscription_expires_at": subscription_expires_at,
        },
        merge=True,
    )
    refreshed = await fs_get("users", user_id)
    if refreshed:
        await sync_user_profile(refreshed)


def entitlement_payload(user: dict) -> dict:
    used = int(user.get("free_prompts_used", 0))
    subscribed = is_user_subscribed(user)
    remaining = None if subscribed else max(0, FREE_PROMPT_LIMIT - used)
    paywall_required = False if subscribed else used >= FREE_PROMPT_LIMIT
    return {
        "free_prompts_used": used,
        "free_prompts_remaining": remaining,
        "paywall_required": paywall_required,
        "subscription_status": "active" if subscribed else str(user.get("subscription_status") or "free"),
        "subscription_expires_at": user.get("subscription_expires_at"),
    }


def paywall_exception(user: dict) -> HTTPException:
    payload = entitlement_payload(user)
    payload["message"] = "Free prompts used. Subscribe to continue."
    return HTTPException(status_code=402, detail=payload)


async def get_usage_today(user_id: str) -> int:
    doc = await fs_get("usage", _fs_doc_id(user_id, today_key()))
    return int(doc["count"]) if doc else 0


async def increment_usage(user_id: str) -> int:
    doc_id = _fs_doc_id(user_id, today_key())
    existing = await fs_get("usage", doc_id)
    res = {
        "id": doc_id,
        "user_id": user_id,
        "date": today_key(),
        "count": int(existing.get("count", 0)) + 1 if existing else 1,
        "created_at": existing.get("created_at", utcnow().isoformat()) if existing else utcnow().isoformat(),
    }
    await fs_set("usage", doc_id, res, merge=False)
    await sync_usage_record(user_id)
    return int(res["count"])


async def enforce_prompt_quota(user_id: str):
    used = await get_usage_today(user_id)
    if used >= DAILY_PROMPT_LIMIT:
        raise HTTPException(status_code=429, detail=f"Daily limit of {DAILY_PROMPT_LIMIT} prompts reached. Try again tomorrow.")


async def reserve_free_prompt(user: dict) -> dict:
    if is_user_subscribed(user):
        refreshed = await fs_get("users", user["id"])
        if refreshed:
            await sync_user_profile(refreshed)
        return refreshed or user

    current = await fs_get("users", user["id"]) or user
    used = int(current.get("free_prompts_used", 0))
    if used >= FREE_PROMPT_LIMIT:
        raise paywall_exception(current)
    refreshed = {**current, "free_prompts_used": used + 1}
    await fs_set("users", user["id"], {"free_prompts_used": used + 1}, merge=True)
    if not refreshed:
        raise paywall_exception(current)
    await sync_user_profile(refreshed)
    return refreshed


async def refund_free_prompt(user_id: str):
    refreshed = await fs_get("users", user_id)
    if refreshed and int(refreshed.get("free_prompts_used", 0)) > 0:
        next_used = int(refreshed.get("free_prompts_used", 0)) - 1
        await fs_set("users", user_id, {"free_prompts_used": next_used}, merge=True)
        refreshed["free_prompts_used"] = next_used
    if refreshed:
        await sync_user_profile(refreshed)


def log_ai_event(
    *,
    user_id: str,
    provider: str,
    model: str,
    prompt_length: int,
    image_count: int,
    success: bool,
    user: dict,
    estimated_cost: Optional[str] = None,
):
    logger.info(
        "ai_event user_id=%s provider=%s model=%s prompt_length=%s image_count=%s success=%s subscription_status=%s free_prompts_used=%s estimated_cost=%s",
        user_id,
        provider,
        model,
        prompt_length,
        image_count,
        success,
        "active" if is_user_subscribed(user) else str(user.get("subscription_status") or "free"),
        int(user.get("free_prompts_used", 0)),
        estimated_cost or "",
    )


async def _firestore_set(path_parts: List[str], payload: dict):
    client = _firestore_db()
    if client is None:
        return

    def _write():
        ref = client
        for index, part in enumerate(path_parts):
            ref = ref.collection(part) if index % 2 == 0 else ref.document(part)
        ref.set(payload, merge=True)

    await asyncio.to_thread(_write)


async def _firestore_delete(path_parts: List[str]):
    client = _firestore_db()
    if client is None:
        return

    def _delete():
        ref = client
        for index, part in enumerate(path_parts):
            ref = ref.collection(part) if index % 2 == 0 else ref.document(part)
        ref.delete()

    await asyncio.to_thread(_delete)


async def sync_user_profile(user_doc: dict):
    await fs_set(
        "users",
        user_doc["id"],
        {
            "id": user_doc["id"],
            "firebase_uid": user_doc.get("firebase_uid"),
            "email": user_doc.get("email"),
            "name": user_doc.get("name"),
            "onboarded": bool(user_doc.get("onboarded", False)),
            "free_prompts_used": int(user_doc.get("free_prompts_used", 0)),
            "subscription_status": user_doc.get("subscription_status", "free"),
            "subscription_expires_at": user_doc.get("subscription_expires_at"),
            "created_at": user_doc.get("created_at"),
            "updated_at": utcnow().isoformat(),
        },
        merge=True,
    )


async def sync_usage_record(user_id: str):
    usage_doc = await fs_get("usage", _fs_doc_id(user_id, today_key()))
    if not usage_doc:
        return
    await _firestore_set(["users", user_id, "usage", usage_doc["date"]], usage_doc)


async def sync_social_account_doc(doc: dict):
    await _firestore_set(["users", doc["user_id"], "social_accounts", doc["platform"]], doc)


async def delete_social_account_doc(user_id: str, platform: str):
    await _firestore_delete(["users", user_id, "social_accounts", platform])


async def sync_chat_session_doc(doc: dict):
    await _firestore_set(["users", doc["user_id"], "chat_sessions", doc["id"]], doc)


async def sync_chat_message_doc(doc: dict):
    await _firestore_set(["users", doc["user_id"], "chat_sessions", doc["session_id"], "messages", doc["id"]], doc)


async def sync_post_doc(doc: dict):
    await _firestore_set(["users", doc["user_id"], "posts", doc["id"]], doc)


async def sync_image_doc(doc: dict):
    await _firestore_set(
        ["users", doc["user_id"], "images", doc["id"]],
        {
            "id": doc["id"],
            "user_id": doc["user_id"],
            "mime": doc.get("mime"),
            "source": doc.get("source"),
            "meta": doc.get("meta", {}),
            "created_at": doc.get("created_at"),
        },
    )


def require_config(value: str, name: str):
    if not value:
        raise HTTPException(status_code=503, detail=f"{name} is not configured on the server yet.")
    return value


def _meta_redirect_with_status(
    app_redirect_uri: str,
    *,
    status_value: str,
    message: str = "",
    platforms: str = "",
    extra: Optional[dict] = None,
):
    separator = "&" if "?" in app_redirect_uri else "?"
    params = {"status": status_value, "message": message, "platforms": platforms}
    if extra:
        params.update({k: v for k, v in extra.items() if v is not None})
    return RedirectResponse(f"{app_redirect_uri}{separator}{urlencode(params)}", status_code=302)


async def _get_json(url: str, *, headers: Optional[dict] = None, timeout: int = 60) -> dict:
    def _do_get():
        response = requests.get(url, headers=headers or {}, timeout=timeout)
        try:
            payload = response.json()
        except Exception:
            payload = {"raw": response.text}
        if response.status_code >= 400:
            detail = payload.get("error", {}).get("message") if isinstance(payload, dict) else None
            raise HTTPException(status_code=502, detail=detail or f"Upstream request failed ({response.status_code})")
        return payload

    return await asyncio.to_thread(_do_get)


async def _post_json(url: str, *, headers: Optional[dict] = None, payload: Optional[dict] = None, timeout: int = 120) -> dict:
    def _do_post():
        response = requests.post(url, headers=headers or {}, json=payload or {}, timeout=timeout)
        try:
            data = response.json()
        except Exception:
            data = {"raw": response.text}
        if response.status_code >= 400:
            detail = data.get("error", {}).get("message") if isinstance(data, dict) else None
            raise HTTPException(status_code=502, detail=detail or f"Upstream post failed ({response.status_code})")
        return data

    return await asyncio.to_thread(_do_post)


async def _post_multipart(
    url: str,
    *,
    headers: Optional[dict] = None,
    data: Optional[dict] = None,
    files: Optional[list[tuple[str, tuple]]] = None,
    timeout: int = 120,
) -> dict:
    def _do_post():
        response = requests.post(url, headers=headers or {}, data=data or {}, files=files or [], timeout=timeout)
        try:
            payload = response.json()
        except Exception:
            payload = {"raw": response.text}
        if response.status_code >= 400:
            detail = payload.get("error", {}).get("message") if isinstance(payload, dict) else None
            raise HTTPException(status_code=502, detail=detail or f"Upstream multipart post failed ({response.status_code})")
        return payload

    return await asyncio.to_thread(_do_post)


async def _get_binary(url: str, *, timeout: int = 120) -> tuple[bytes, str]:
    def _do_get():
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.content, (response.headers.get("Content-Type") or "").split(";")[0].strip()

    try:
        return await asyncio.to_thread(_do_get)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Could not download generated image: {exc}") from exc


def _extract_result_urls(result_json) -> list[str]:
    if isinstance(result_json, str):
        try:
            result_json = json.loads(result_json)
        except Exception:
            return []
    if isinstance(result_json, dict):
        if isinstance(result_json.get("resultUrls"), list):
            return [str(item) for item in result_json["resultUrls"] if item]
        if isinstance(result_json.get("images"), list):
            return [str(item) for item in result_json["images"] if item]
    if isinstance(result_json, list):
        return [str(item) for item in result_json if item]
    return []


def _extract_kie_task_id(payload: dict | None) -> str | None:
    if not isinstance(payload, dict):
        return None
    for key in ("taskId", "task_id"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    data = payload.get("data")
    if isinstance(data, dict):
        for key in ("taskId", "task_id"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _extract_kie_error(payload: dict | None) -> str | None:
    if not isinstance(payload, dict):
        return None
    for key in ("msg", "message", "error", "detail"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    data = payload.get("data")
    if isinstance(data, dict):
        for key in ("msg", "message", "error", "detail"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


async def _upload_kie_base64_image(image_b64: str, mime_type: str) -> str:
    require_config(KIE_API_KEY, "KIE_API_KEY")
    extension = mimetypes.guess_extension(mime_type or "") or ".png"
    data = await _post_json(
        f"{KIE_UPLOAD_BASE_URL}/api/file-base64-upload",
        headers={
            "Authorization": f"Bearer {KIE_API_KEY}",
            "Content-Type": "application/json",
        },
        payload={
            "base64Data": f"data:{mime_type};base64,{image_b64}",
            "uploadPath": "images/evolve",
            "fileName": f"upload-{uuid.uuid4().hex}{extension}",
        },
        timeout=60,
    )
    file_data = (data or {}).get("data") or {}
    download_url = file_data.get("downloadUrl") or file_data.get("fileUrl")
    if not download_url:
        raise HTTPException(status_code=502, detail="Image upload provider did not return a file URL.")
    return download_url


async def _create_kie_image_task(prompt: str, image_b64: str, mime_type: str) -> str:
    require_config(KIE_API_KEY, "KIE_API_KEY")
    source_image_url = await _upload_kie_base64_image(image_b64, mime_type)
    payload = {
        "model": KIE_IMAGE_MODEL,
        "input": {
            "prompt": prompt,
            "image_input": [source_image_url],
            "aspect_ratio": KIE_IMAGE_ASPECT_RATIO,
            "resolution": KIE_IMAGE_RESOLUTION,
            "output_format": KIE_IMAGE_FORMAT,
        },
    }
    data = await _post_json(
        f"{KIE_API_BASE_URL}/api/v1/jobs/createTask",
        headers={"Authorization": f"Bearer {KIE_API_KEY}"},
        payload=payload,
        timeout=60,
    )
    task_id = _extract_kie_task_id(data)
    if not task_id:
        provider_error = _extract_kie_error(data)
        if provider_error:
            raise HTTPException(status_code=502, detail=f"Image provider error: {provider_error}")
        raise HTTPException(status_code=502, detail="Image provider did not return a task ID.")
    return task_id


async def _wait_for_kie_image(task_id: str) -> tuple[str, str]:
    deadline = asyncio.get_running_loop().time() + KIE_POLL_TIMEOUT_SECONDS
    headers = {"Authorization": f"Bearer {KIE_API_KEY}"}
    while True:
        data = await _get_json(
            f"{KIE_API_BASE_URL}/api/v1/jobs/recordInfo?taskId={task_id}",
            headers=headers,
            timeout=60,
        )
        job = (data or {}).get("data") or {}
        state = job.get("state")
        if state == "success":
            urls = _extract_result_urls(job.get("resultJson"))
            if not urls:
                raise HTTPException(status_code=502, detail="Image provider finished without returning an image URL.")
            image_bytes, content_type = await _get_binary(urls[0])
            mime_type = content_type or mimetypes.guess_type(urls[0])[0] or "image/png"
            return base64.b64encode(image_bytes).decode("utf-8"), mime_type
        if state == "fail":
            raise HTTPException(status_code=502, detail=f"Image provider error: {job.get('failMsg') or 'Image generation failed.'}")
        if asyncio.get_running_loop().time() >= deadline:
            raise HTTPException(status_code=504, detail="Image generation timed out.")
        await asyncio.sleep(KIE_POLL_INTERVAL_SECONDS)


async def _generate_kie_image(prompt: str, image_b64: str, mime_type: str) -> tuple[str, str]:
    task_id = await _create_kie_image_task(prompt, image_b64, mime_type)
    return await _wait_for_kie_image(task_id)


async def _openai_responses_text(prompt_text: str, image_b64: str, mime_type: str, system_text: str) -> str:
    require_config(OPENAI_API_KEY, "OPENAI_API_KEY")
    data = await _post_json(
        f"{OPENAI_API_BASE_URL}/responses",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        payload={
            "model": OPENAI_TEXT_MODEL,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_text}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt_text},
                        {"type": "input_image", "image_url": f"data:{mime_type};base64,{image_b64}"},
                    ],
                },
            ],
        },
        timeout=90,
    )
    output_text = data.get("output_text")
    if output_text:
        return output_text.strip()
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                return str(content["text"]).strip()
    raise HTTPException(status_code=502, detail="OpenAI returned an unexpected response.")


def _meta_headers(access_token: str) -> dict:
    return {"Authorization": f"Bearer {access_token}"}


async def _meta_get(path: str, *, access_token: str, params: Optional[dict] = None) -> dict:
    query = urlencode(params or {})
    url = f"{META_GRAPH_API_BASE_URL}{path}"
    if query:
        url = f"{url}?{query}"
    return await _get_json(url, headers=_meta_headers(access_token), timeout=60)


async def _exchange_meta_code_for_user_token(code: str) -> str:
    require_config(META_APP_ID, "META_APP_ID")
    require_config(META_APP_SECRET, "META_APP_SECRET")
    require_config(META_REDIRECT_URI, "META_REDIRECT_URI")
    params = urlencode(
        {
            "client_id": META_APP_ID,
            "client_secret": META_APP_SECRET,
            "redirect_uri": META_REDIRECT_URI,
            "code": code,
        }
    )
    token_data = await _get_json(f"{META_GRAPH_API_BASE_URL}/oauth/access_token?{params}", timeout=60)
    user_token = token_data.get("access_token")
    if not user_token:
        raise HTTPException(status_code=502, detail="Meta did not return a user access token.")

    exchange_params = urlencode(
        {
            "grant_type": "fb_exchange_token",
            "client_id": META_APP_ID,
            "client_secret": META_APP_SECRET,
            "fb_exchange_token": user_token,
        }
    )
    exchange_data = await _get_json(f"{META_GRAPH_API_BASE_URL}/oauth/access_token?{exchange_params}", timeout=60)
    return exchange_data.get("access_token") or user_token


async def _upsert_social_connection(
    *,
    user_id: str,
    platform: str,
    account_name: str,
    account_id: str,
    access_token: str,
    metadata: Optional[dict] = None,
):
    existing = await fs_get("social_accounts", _fs_doc_id(user_id, platform))
    doc = {
        "id": existing["id"] if existing else _fs_doc_id(user_id, platform),
        "user_id": user_id,
        "platform": platform,
        "handle": account_name,
        "account_id": account_id,
        "access_token": access_token,
        "metadata": metadata or {},
        "status": "connected",
        "connected_at": existing.get("connected_at", utcnow().isoformat()) if existing else utcnow().isoformat(),
        "updated_at": utcnow().isoformat(),
    }
    await fs_set("social_accounts", doc["id"], doc, merge=False)
    await sync_social_account_doc(doc)
    return doc


async def _publish_facebook_photo(*, page_id: str, access_token: str, image_url: str, caption_text: str) -> str:
    payload = {"url": image_url, "caption": caption_text, "published": True}
    data = await _post_json(
        f"{META_GRAPH_API_BASE_URL}/{page_id}/photos?access_token={access_token}",
        headers={"Content-Type": "application/json"},
        payload=payload,
        timeout=120,
    )
    post_id = data.get("post_id") or data.get("id")
    if not post_id:
        raise HTTPException(status_code=502, detail="Meta did not return a Facebook post ID.")
    return post_id


async def _publish_instagram_photo(*, instagram_account_id: str, access_token: str, image_url: str, caption_text: str) -> str:
    create_data = await _post_json(
        f"{META_GRAPH_API_BASE_URL}/{instagram_account_id}/media?access_token={access_token}",
        headers={"Content-Type": "application/json"},
        payload={"image_url": image_url, "caption": caption_text},
        timeout=120,
    )
    creation_id = create_data.get("id")
    if not creation_id:
        raise HTTPException(status_code=502, detail="Meta did not return an Instagram media container ID.")
    publish_data = await _post_json(
        f"{META_GRAPH_API_BASE_URL}/{instagram_account_id}/media_publish?access_token={access_token}",
        headers={"Content-Type": "application/json"},
        payload={"creation_id": creation_id},
        timeout=120,
    )
    media_id = publish_data.get("id")
    if not media_id:
        raise HTTPException(status_code=502, detail="Meta did not return an Instagram media ID.")
    return media_id


# ---------- Abuse / cost protection ----------
_user_locks: dict[str, asyncio.Lock] = {}
_user_call_times: dict[str, deque] = defaultdict(deque)
_ip_register_times: dict[str, deque] = defaultdict(deque)


def _user_lock(user_id: str) -> asyncio.Lock:
    lock = _user_locks.get(user_id)
    if lock is None:
        lock = asyncio.Lock()
        _user_locks[user_id] = lock
    return lock


def _enforce_rate_limit(user_id: str):
    now = utcnow().timestamp()
    bucket = _user_call_times[user_id]
    # Drop calls older than 60s
    while bucket and now - bucket[0] > 60:
        bucket.popleft()
    if len(bucket) >= MAX_AI_PER_MINUTE_PER_USER:
        retry_in = int(60 - (now - bucket[0])) + 1
        raise HTTPException(
            status_code=429,
            detail=f"You're going a bit fast — wait {retry_in}s before the next AI request.",
        )
    bucket.append(now)


async def _enforce_global_cap():
    res = await fs_get("global_usage", today_key())
    used = int(res.get("count", 0)) if res else 0
    if used >= GLOBAL_DAILY_AI_CAP:
        raise HTTPException(
            status_code=503,
            detail="Daily AI capacity reached for today. We're protecting service quality — try again tomorrow.",
        )


async def _increment_global():
    doc_id = today_key()
    existing = await fs_get("global_usage", doc_id)
    await fs_set(
        "global_usage",
        doc_id,
        {
            "id": doc_id,
            "date": doc_id,
            "count": int(existing.get("count", 0)) + 1 if existing else 1,
            "created_at": existing.get("created_at", utcnow().isoformat()) if existing else utcnow().isoformat(),
        },
        merge=False,
    )


async def _decrement_user_usage(user_id: str):
    """Refund a prompt if the AI call failed AFTER the reservation."""
    doc_id = _fs_doc_id(user_id, today_key())
    existing = await fs_get("usage", doc_id)
    if existing and int(existing.get("count", 0)) > 0:
        await fs_set("usage", doc_id, {"count": int(existing.get("count", 0)) - 1}, merge=True)
    await sync_usage_record(user_id)


async def _decrement_global():
    existing = await fs_get("global_usage", today_key())
    if existing and int(existing.get("count", 0)) > 0:
        await fs_set("global_usage", today_key(), {"count": int(existing.get("count", 0)) - 1}, merge=True)


def _client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for") or request.headers.get("X-Forwarded-For")
    if xff:
        return xff.split(",")[0].strip()
    xri = request.headers.get("x-real-ip") or request.headers.get("X-Real-IP")
    if xri:
        return xri.strip()
    return (request.client.host if request.client else "unknown") or "unknown"


def _enforce_register_ip_rate(ip: str):
    now = utcnow().timestamp()
    bucket = _ip_register_times[ip]
    while bucket and now - bucket[0] > 3600:
        bucket.popleft()
    if len(bucket) >= REGISTRATIONS_PER_HOUR_PER_IP:
        raise HTTPException(status_code=429, detail="Too many signups from this network. Try again later.")
    bucket.append(now)


def _normalize_image(raw: bytes, mime_hint: str) -> tuple[bytes, str]:
    """Open, strip animation, downscale, re-encode as JPEG/PNG/WEBP. Caps size + dimensions."""
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Image too large (max {MAX_UPLOAD_BYTES // (1024*1024)}MB).")
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image.")
    # First frame only (animated GIF/WEBP)
    if getattr(img, "is_animated", False):
        try:
            img.seek(0)
        except Exception:
            pass
    # Convert mode for safety
    if img.mode in ("P", "RGBA", "LA"):
        img = img.convert("RGBA")
    else:
        img = img.convert("RGB")
    # Downscale longest side
    w, h = img.size
    longest = max(w, h)
    if longest > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / longest
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    # Encode: prefer JPEG for size; keep PNG if has alpha
    out = io.BytesIO()
    if img.mode == "RGBA":
        img.save(out, format="PNG", optimize=True)
        out_mime = "image/png"
    else:
        img.save(out, format="JPEG", quality=88, optimize=True)
        out_mime = "image/jpeg"
    data = out.getvalue()
    if len(data) > MAX_UPLOAD_BYTES:
        # Hard-clip with lower quality re-encode
        out = io.BytesIO()
        img.convert("RGB").save(out, format="JPEG", quality=72, optimize=True)
        data = out.getvalue()
        out_mime = "image/jpeg"
    return data, out_mime


# ---------- Models ----------
class UserPublic(BaseModel):
    id: str
    email: EmailStr
    name: str
    onboarded: bool = False
    free_prompts_used: int = 0
    subscription_status: str = "free"
    subscription_expires_at: Optional[str] = None
    created_at: str


class RegisterBody(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6)
    name: str = Field(min_length=1, max_length=80)


class LoginBody(BaseModel):
    email: EmailStr
    password: str


class FirebaseAuthBody(BaseModel):
    id_token: str
    name: Optional[str] = None


class AuthResponse(BaseModel):
    token: str
    user: UserPublic


class RestorePurchaseBody(BaseModel):
    receipt: Optional[str] = None


class SubscriptionStatusBody(BaseModel):
    subscription_status: str
    subscription_expires_at: Optional[str] = None


class OnboardingBody(BaseModel):
    onboarded: bool = True


class SocialConnectBody(BaseModel):
    platform: Literal["instagram", "facebook"]
    handle: str


class MetaConnectionSelectionIn(BaseModel):
    page_id: str


class ChatSessionCreate(BaseModel):
    title: Optional[str] = None


class ChatSession(BaseModel):
    id: str
    user_id: str
    title: str
    created_at: str
    updated_at: str
    cover_image_id: Optional[str] = None


class EditImageBody(BaseModel):
    session_id: str
    image_id: Optional[str] = None
    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
    provider: Literal["kie", "gemini", "openai"] = "kie"


class CaptionBody(BaseModel):
    image_id: str
    style: Optional[str] = Field(default="instagram", max_length=40)
    extra_instructions: Optional[str] = Field(default="", max_length=MAX_PROMPT_CHARS)


class PublishBody(BaseModel):
    image_id: str
    caption: str
    platform: Literal["instagram", "facebook"] = "instagram"
    session_id: Optional[str] = None


class ChatMessage(BaseModel):
    id: str
    session_id: str
    user_id: str
    role: Literal["user", "assistant", "system"]
    kind: Literal["text", "image", "caption", "suggestions", "publish"]
    content: str = ""
    image_id: Optional[str] = None
    suggestions: Optional[List[str]] = None
    meta: Optional[dict] = None
    created_at: str


# ---------- Health ----------
@api_router.get("/")
async def root():
    return {"ok": True, "service": "Evolve Content Creator API"}


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled server error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
    )


# ---------- Auth ----------
@api_router.post("/auth/register", response_model=AuthResponse)
async def register(body: RegisterBody, request: Request):
    raise HTTPException(status_code=410, detail="Use Firebase Auth on the client to register.")


@api_router.post("/auth/login", response_model=AuthResponse)
async def login(body: LoginBody):
    raise HTTPException(status_code=410, detail="Use Firebase Auth on the client to log in.")


@api_router.post("/auth/firebase", response_model=AuthResponse)
async def firebase_auth_exchange(body: FirebaseAuthBody):
    if _initialize_firebase() is None:
        raise HTTPException(status_code=503, detail="Firebase Admin is not configured on the server yet.")
    try:
        decoded = await asyncio.to_thread(firebase_auth_admin.verify_id_token, body.id_token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid Firebase ID token")

    firebase_uid = decoded.get("uid")
    email = (decoded.get("email") or "").strip().lower()
    if not firebase_uid or not email:
        raise HTTPException(status_code=400, detail="Firebase token did not contain a valid uid/email")

    name = (
        body.name
        or decoded.get("name")
        or email.split("@")[0]
    ).strip()[:80]

    user = await fs_user_upsert_from_firebase(firebase_uid, email, name)
    await sync_user_profile(user)
    return AuthResponse(token=body.id_token, user=UserPublic(**user))


@api_router.get("/auth/me", response_model=UserPublic)
async def me(user: dict = Depends(get_current_user)):
    return UserPublic(**user)


@api_router.post("/auth/onboarding", response_model=UserPublic)
async def complete_onboarding(body: OnboardingBody, user: dict = Depends(get_current_user)):
    await fs_set("users", user["id"], {"onboarded": body.onboarded}, merge=True)
    user["onboarded"] = body.onboarded
    await sync_user_profile(user)
    return UserPublic(**user)


@api_router.get("/billing/status")
async def billing_status(user: dict = Depends(get_current_user)):
    return entitlement_payload(user)


@api_router.post("/billing/restore")
async def restore_purchase(body: RestorePurchaseBody, user: dict = Depends(get_current_user)):
    valid = await verify_subscription_receipt(body.receipt, user)
    if not valid:
        return {"restored": False, **entitlement_payload(user)}
    await update_subscription_status(user["id"], "active")
    refreshed = await fs_get("users", user["id"]) or user
    return {"restored": True, **entitlement_payload(refreshed)}


@api_router.post("/billing/update-status")
async def update_billing_status(body: SubscriptionStatusBody, user: dict = Depends(get_current_user)):
    await update_subscription_status(
        user["id"],
        body.subscription_status,
        body.subscription_expires_at,
    )
    refreshed = await fs_get("users", user["id"]) or user
    return entitlement_payload(refreshed)


# ---------- Usage ----------
@api_router.get("/usage/today")
async def usage_today(user: dict = Depends(get_current_user)):
    used = await get_usage_today(user["id"])
    return {
        "used": used,
        "limit": DAILY_PROMPT_LIMIT,
        "remaining": max(0, DAILY_PROMPT_LIMIT - used),
        "date": today_key(),
        **entitlement_payload(user),
    }


# ---------- Social Accounts ----------
@api_router.get("/social/accounts")
async def list_social(user: dict = Depends(get_current_user)):
    docs = await fs_query("social_accounts", filters=[("user_id", "==", user["id"])], limit=50)
    return docs


@api_router.post("/social/accounts")
async def connect_social(body: SocialConnectBody, user: dict = Depends(get_current_user)):
    sid = _fs_doc_id(user["id"], body.platform)
    doc = {
        "id": sid,
        "user_id": user["id"],
        "platform": body.platform,
        "handle": body.handle.lstrip("@").strip(),
        "connected_at": utcnow().isoformat(),
        "status": "connected_mock",
    }
    await fs_set("social_accounts", sid, doc, merge=False)
    await sync_social_account_doc(doc)
    return doc


@api_router.get("/social/meta/start")
async def start_meta_connect(
    platform: Literal["instagram", "facebook"],
    app_redirect_uri: Optional[str] = None,
    user: dict = Depends(get_current_user),
):
    require_config(META_APP_ID, "META_APP_ID")
    require_config(META_APP_SECRET, "META_APP_SECRET")
    require_config(META_REDIRECT_URI, "META_REDIRECT_URI")
    state = str(uuid.uuid4())
    redirect_target = (app_redirect_uri or META_APP_REDIRECT_URI).strip() or META_APP_REDIRECT_URI
    await fs_set("oauth_states", state, {
        "id": state,
        "state": state,
        "user_id": user["id"],
        "platform": platform,
        "app_redirect_uri": redirect_target,
        "created_at": utcnow().isoformat(),
    }, merge=False)
    scopes = [
        "pages_show_list",
        "pages_read_engagement",
        "pages_manage_posts",
        "instagram_basic",
        "instagram_content_publish",
        "business_management",
    ]
    auth_url = f"{META_OAUTH_BASE_URL}/dialog/oauth?" + urlencode(
        {
            "client_id": META_APP_ID,
            "redirect_uri": META_REDIRECT_URI,
            "state": state,
            "response_type": "code",
            "scope": ",".join(scopes),
        }
    )
    return {"auth_url": auth_url, "state": state}


@api_router.get("/oauth/meta/callback")
async def meta_oauth_callback(
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
    error_message: Optional[str] = None,
):
    state_doc = await fs_get("oauth_states", state) if state else None
    app_redirect = (state_doc or {}).get("app_redirect_uri") or META_APP_REDIRECT_URI
    if error:
        return _meta_redirect_with_status(app_redirect, status_value="error", message=error_message or error)
    if not state_doc:
        return _meta_redirect_with_status(app_redirect, status_value="error", message="Missing or invalid OAuth state")
    if not code:
        await fs_delete("oauth_states", state)
        return _meta_redirect_with_status(app_redirect, status_value="error", message="Missing authorization code")
    try:
        access_token = await _exchange_meta_code_for_user_token(code)
        profile = await _meta_get("/me", access_token=access_token, params={"fields": "id,name,email"})
        pages = await _meta_get("/me/accounts", access_token=access_token, params={"fields": "id,name,access_token"})
        page_rows = pages.get("data") or []
        options = []
        for facebook_page in page_rows:
            page_token = facebook_page.get("access_token") or access_token
            page_detail = await _meta_get(
                f"/{facebook_page['id']}",
                access_token=page_token,
                params={"fields": "id,name,instagram_business_account{id,username,name}"},
            )
            options.append(
                {
                    "page_id": facebook_page["id"],
                    "page_name": facebook_page.get("name") or "Facebook Page",
                    "page_access_token": page_token,
                    "instagram_account": page_detail.get("instagram_business_account"),
                }
            )
        selection_id = str(uuid.uuid4())
        await fs_set("meta_connection_options", selection_id, {
            "id": selection_id,
            "user_id": state_doc["user_id"],
            "requested_platform": state_doc.get("platform", "facebook"),
            "profile": profile,
            "page_options": options,
            "access_token": access_token,
            "created_at": utcnow().isoformat(),
        }, merge=False)
        await fs_delete("oauth_states", state)
        return _meta_redirect_with_status(
            app_redirect,
            status_value="needs_selection",
            message="Choose the page/account to connect.",
            extra={"selection_id": selection_id, "requested_platform": state_doc.get("platform", "facebook")},
        )
    except HTTPException as exc:
        await fs_delete("oauth_states", state)
        return _meta_redirect_with_status(app_redirect, status_value="error", message=str(exc.detail))
    except Exception as exc:
        await fs_delete("oauth_states", state)
        return _meta_redirect_with_status(app_redirect, status_value="error", message=f"Meta connection failed: {exc}")


@api_router.get("/social/meta/options/{selection_id}")
async def get_meta_connection_options(selection_id: str, user: dict = Depends(get_current_user)):
    doc = await fs_get("meta_connection_options", selection_id)
    if doc and doc.get("user_id") != user["id"]:
        doc = None
    if doc:
        doc.pop("access_token", None)
    if not doc:
        raise HTTPException(status_code=404, detail="Meta connection selection not found")
    return doc


@api_router.post("/social/meta/options/{selection_id}/select")
async def select_meta_connection_option(selection_id: str, data: MetaConnectionSelectionIn, user: dict = Depends(get_current_user)):
    doc = await fs_get("meta_connection_options", selection_id)
    if doc and doc.get("user_id") != user["id"]:
        doc = None
    if not doc:
        raise HTTPException(status_code=404, detail="Meta connection selection not found")
    option = next((item for item in (doc.get("page_options") or []) if item.get("page_id") == data.page_id), None)
    if not option:
        raise HTTPException(status_code=404, detail="Selected Meta page was not found")
    connected = []
    await _upsert_social_connection(
        user_id=user["id"],
        platform="facebook",
        account_name=option.get("page_name") or "Facebook Page",
        account_id=option["page_id"],
        access_token=option.get("page_access_token") or doc.get("access_token", ""),
        metadata={
            "meta_user_id": (doc.get("profile") or {}).get("id"),
            "meta_user_name": (doc.get("profile") or {}).get("name"),
            "meta_user_email": (doc.get("profile") or {}).get("email"),
        },
    )
    connected.append("facebook")
    instagram_account = option.get("instagram_account")
    if instagram_account:
        await _upsert_social_connection(
            user_id=user["id"],
            platform="instagram",
            account_name=instagram_account.get("username") or instagram_account.get("name") or "Instagram Business",
            account_id=instagram_account["id"],
            access_token=option.get("page_access_token") or doc.get("access_token", ""),
            metadata={
                "facebook_page_id": option["page_id"],
                "facebook_page_name": option.get("page_name"),
            },
        )
        connected.append("instagram")
    await fs_delete("meta_connection_options", selection_id)
    return {"connected_platforms": connected}


@api_router.delete("/social/accounts/{platform}")
async def disconnect_social(platform: str, user: dict = Depends(get_current_user)):
    await fs_delete("social_accounts", _fs_doc_id(user["id"], platform))
    await delete_social_account_doc(user["id"], platform)
    return {"ok": True}


# ---------- Images ----------
async def save_image_record(user_id: str, data_b64: str, mime: str, source: str, meta: Optional[dict] = None) -> dict:
    img_id = str(uuid.uuid4())
    doc = {
        "id": img_id,
        "user_id": user_id,
        "mime": mime,
        "data_b64": data_b64,
        "source": source,  # 'upload' | 'gemini' | 'openai'
        "meta": meta or {},
        "created_at": utcnow().isoformat(),
    }
    await fs_set("images", img_id, doc, merge=False)
    await sync_image_doc(doc)
    return {"id": img_id, "mime": mime, "source": source, "url": f"/api/images/{img_id}"}


@api_router.post("/images/upload")
async def upload_image(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    raw = await file.read()
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    mime = (file.content_type or "image/png").lower()
    if mime not in ("image/png", "image/jpeg", "image/jpg", "image/webp"):
        raise HTTPException(status_code=400, detail=f"Unsupported image type: {mime}")
    # Normalize: enforce max size, downscale, strip animation, re-encode
    data, out_mime = _normalize_image(raw, mime)
    b64 = base64.b64encode(data).decode("utf-8")
    rec = await save_image_record(user["id"], b64, out_mime, "upload")
    return rec


@api_router.get("/images/{image_id}")
async def get_image(image_id: str):
    doc = await fs_get("images", image_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Image not found")
    raw = base64.b64decode(doc["data_b64"])
    return Response(content=raw, media_type=doc.get("mime", "image/png"))


# ---------- Chat Sessions ----------
@api_router.post("/chat/sessions", response_model=ChatSession)
async def create_session(body: ChatSessionCreate, user: dict = Depends(get_current_user)):
    sid = str(uuid.uuid4())
    now = utcnow().isoformat()
    doc = {
        "id": sid,
        "user_id": user["id"],
        "title": (body.title or "New creation").strip()[:80],
        "created_at": now,
        "updated_at": now,
        "cover_image_id": None,
    }
    await fs_set("chat_sessions", sid, doc, merge=False)
    await sync_chat_session_doc(doc)
    return ChatSession(**doc)


@api_router.get("/chat/sessions")
async def list_sessions(user: dict = Depends(get_current_user)):
    docs = await fs_query("chat_sessions", filters=[("user_id", "==", user["id"])], order_by="updated_at", descending=True, limit=100)
    return docs


@api_router.get("/chat/sessions/{session_id}")
async def get_session(session_id: str, user: dict = Depends(get_current_user)):
    sess = await fs_get("chat_sessions", session_id)
    if sess and sess.get("user_id") != user["id"]:
        sess = None
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    msgs = await fs_query("chat_messages", filters=[("session_id", "==", session_id), ("user_id", "==", user["id"])], order_by="created_at", limit=1000)
    return {"session": sess, "messages": msgs}


@api_router.delete("/chat/sessions/{session_id}")
async def delete_session(session_id: str, user: dict = Depends(get_current_user)):
    sess = await fs_get("chat_sessions", session_id)
    if sess and sess.get("user_id") == user["id"]:
        await fs_delete("chat_sessions", session_id)
    msgs = await fs_query("chat_messages", filters=[("session_id", "==", session_id), ("user_id", "==", user["id"])], limit=2000)
    for msg in msgs:
        await fs_delete("chat_messages", msg["id"])
    return {"ok": True}


async def push_message(session_id: str, user_id: str, role: str, kind: str, content: str = "",
                       image_id: Optional[str] = None, suggestions: Optional[List[str]] = None,
                       meta: Optional[dict] = None) -> dict:
    msg = {
        "id": str(uuid.uuid4()),
        "session_id": session_id,
        "user_id": user_id,
        "role": role,
        "kind": kind,
        "content": content,
        "image_id": image_id,
        "suggestions": suggestions,
        "meta": meta or {},
        "created_at": utcnow().isoformat(),
    }
    await fs_set("chat_messages", msg["id"], msg, merge=False)
    await fs_set("chat_sessions", session_id, {"updated_at": msg["created_at"]}, merge=True)
    await sync_chat_message_doc(msg)
    session_doc = await fs_get("chat_sessions", session_id)
    if session_doc:
        await sync_chat_session_doc(session_doc)
    return msg


# ---------- AI: Upload image to chat ----------
class AttachImageBody(BaseModel):
    session_id: str
    image_id: str


SUGGESTION_PRESETS = [
    "Make this image more luxury",
    "Turn this into a product ad",
    "Create a caption for Instagram",
    "Make this more viral",
    "Generate 5 post ideas from this image",
    "Add cinematic lighting and depth",
    "Apply a minimalist editorial aesthetic",
]


@api_router.post("/chat/attach-image")
async def attach_image(body: AttachImageBody, user: dict = Depends(get_current_user)):
    sess = await fs_get("chat_sessions", body.session_id)
    if sess and sess.get("user_id") != user["id"]:
        sess = None
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    img = await fs_get("images", body.image_id)
    if img and img.get("user_id") != user["id"]:
        img = None
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")
    # Save user message (image)
    user_msg = await push_message(body.session_id, user["id"], "user", "image",
                                  content="Uploaded image", image_id=body.image_id)
    # Assistant suggestion message
    asst_msg = await push_message(
        body.session_id, user["id"], "assistant", "suggestions",
        content="Great! Here are a few directions we can take this. Pick one or write your own prompt.",
        suggestions=SUGGESTION_PRESETS[:5],
    )
    await fs_set("chat_sessions", body.session_id, {"cover_image_id": body.image_id}, merge=True)
    session_doc = await fs_get("chat_sessions", body.session_id)
    if session_doc:
        await sync_chat_session_doc(session_doc)
    return {"user_message": user_msg, "assistant_message": asst_msg}


@api_router.post("/chat/edit-image")
async def edit_image(body: EditImageBody, user: dict = Depends(get_current_user)):
    session_id = body.session_id
    image_id = body.image_id
    prompt = body.prompt.strip()

    sess = await fs_get("chat_sessions", session_id)
    if sess and sess.get("user_id") != user["id"]:
        sess = None
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")

    # Per-user lock so a user can only have ONE AI call in flight
    lock = _user_lock(user["id"])
    if lock.locked():
        raise HTTPException(status_code=429, detail="An AI request is already running. Please wait.")

    async with lock:
        _enforce_rate_limit(user["id"])
        await enforce_prompt_quota(user["id"])
        await _enforce_global_cap()

        # User message saved early so it shows even if AI fails
        await push_message(session_id, user["id"], "user", "text", content=prompt)

        image_b64 = None
        if image_id:
            img = await fs_get("images", image_id)
            if img and img.get("user_id") != user["id"]:
                img = None
            if not img:
                raise HTTPException(status_code=404, detail="Image not found")
            image_b64 = img["data_b64"]

        # Reserve free entitlement and quota before the expensive AI call.
        user = await reserve_free_prompt(user)
        await increment_usage(user["id"])
        await _increment_global()

        try:
            if not image_b64:
                raise HTTPException(status_code=400, detail="Image editing requires an uploaded source image.")
            source_mime = img.get("mime", "image/png") if image_id and img else "image/png"
            new_b64, new_mime = await _generate_kie_image(prompt, image_b64, source_mime)
        except Exception as e:
            # Refund on failure
            await refund_free_prompt(user["id"])
            await _decrement_user_usage(user["id"])
            await _decrement_global()
            logger.exception("Gemini image edit failed")
            log_ai_event(
                user_id=user["id"],
                provider="kie",
                model=KIE_IMAGE_MODEL,
                prompt_length=len(prompt),
                image_count=1 if image_b64 else 0,
                success=False,
                user=user,
            )
            raise HTTPException(status_code=502, detail=f"AI generation failed: {str(e)[:200]}")

        if not new_b64:
            await refund_free_prompt(user["id"])
            await _decrement_user_usage(user["id"])
            await _decrement_global()
            log_ai_event(
                user_id=user["id"],
                provider="kie",
                model=KIE_IMAGE_MODEL,
                prompt_length=len(prompt),
                image_count=1 if image_b64 else 0,
                success=False,
                user=user,
            )
            raise HTTPException(status_code=502, detail="Empty image from AI")

        new_rec = await save_image_record(user["id"], new_b64, new_mime, "kie",
                                          meta={"prompt": prompt, "session_id": session_id})

        asst_msg = await push_message(
            session_id, user["id"], "assistant", "image",
            content="Here's your refined image. Want to tweak it further or generate a caption?",
            image_id=new_rec["id"],
            suggestions=["Generate a caption for this", "Make it even more dramatic", "Try a different style"],
            meta={"prompt": prompt},
        )
        await fs_set("chat_sessions", session_id, {"cover_image_id": new_rec["id"]}, merge=True)
        session_doc = await fs_get("chat_sessions", session_id)
        if session_doc:
            await sync_chat_session_doc(session_doc)

        refreshed = await fs_get("users", user["id"]) or user
        log_ai_event(
            user_id=user["id"],
            provider="kie",
            model=KIE_IMAGE_MODEL,
            prompt_length=len(prompt),
            image_count=1 if image_b64 else 0,
            success=True,
            user=refreshed,
        )

        return {"message": asst_msg, "image": new_rec, **entitlement_payload(refreshed)}


# ---------- AI: Generate caption ----------
@api_router.post("/chat/caption")
async def generate_caption(body: CaptionBody, user: dict = Depends(get_current_user)):
    img = await fs_get("images", body.image_id)
    if img and img.get("user_id") != user["id"]:
        img = None
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")

    lock = _user_lock(user["id"])
    if lock.locked():
        raise HTTPException(status_code=429, detail="An AI request is already running. Please wait.")

    async with lock:
        _enforce_rate_limit(user["id"])
        await enforce_prompt_quota(user["id"])
        await _enforce_global_cap()
        user = await reserve_free_prompt(user)
        await increment_usage(user["id"])
        await _increment_global()

        system_msg = (
            "You are a world-class social media copywriter. Generate a single high-converting caption "
            "for the provided image. Match the requested platform style, include 3-6 relevant hashtags "
            "on a new line, and keep it natural, engaging, and free of generic AI fluff."
        )

        instr = body.extra_instructions or ""
        style = body.style or "instagram"
        user_text = (
            f"Write a caption for this image. Platform: {style}.\n"
            f"Tone: premium, confident, attention-grabbing first line.\n"
            f"Additional instructions: {instr or 'none'}.\n"
            f"Return only the caption text (with hashtags at the end)."
        )

        try:
            caption = await _openai_responses_text(user_text, img["data_b64"], img.get("mime", "image/png"), system_msg)
        except Exception as e:
            await refund_free_prompt(user["id"])
            await _decrement_user_usage(user["id"])
            await _decrement_global()
            logger.exception("Caption generation failed")
            log_ai_event(
                user_id=user["id"],
                provider="openai",
                model=OPENAI_TEXT_MODEL,
                prompt_length=len(user_text),
                image_count=1,
                success=False,
                user=user,
            )
            raise HTTPException(status_code=502, detail=f"Caption generation failed: {str(e)[:200]}")

        refreshed = await fs_get("users", user["id"]) or user
        log_ai_event(
            user_id=user["id"],
            provider="openai",
            model=OPENAI_TEXT_MODEL,
            prompt_length=len(user_text),
            image_count=1,
            success=True,
            user=refreshed,
        )

        return {"caption": caption, "style": style, "image_id": body.image_id, **entitlement_payload(refreshed)}


class SaveCaptionMessageBody(BaseModel):
    session_id: str
    image_id: str
    caption: str


@api_router.post("/chat/caption-message")
async def save_caption_message(body: SaveCaptionMessageBody, user: dict = Depends(get_current_user)):
    sess = await fs_get("chat_sessions", body.session_id)
    if sess and sess.get("user_id") != user["id"]:
        sess = None
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    msg = await push_message(
        body.session_id, user["id"], "assistant", "caption",
        content=body.caption, image_id=body.image_id,
        suggestions=["Approve & publish", "Regenerate caption", "Edit caption"],
    )
    return msg


# ---------- AI: Suggestions / Post ideas ----------
class SuggestBody(BaseModel):
    image_id: str
    session_id: Optional[str] = None


@api_router.post("/chat/post-ideas")
async def generate_post_ideas(body: SuggestBody, user: dict = Depends(get_current_user)):
    img = await fs_get("images", body.image_id)
    if img and img.get("user_id") != user["id"]:
        img = None
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")

    lock = _user_lock(user["id"])
    if lock.locked():
        raise HTTPException(status_code=429, detail="An AI request is already running. Please wait.")

    async with lock:
        _enforce_rate_limit(user["id"])
        await enforce_prompt_quota(user["id"])
        await _enforce_global_cap()

        user = await reserve_free_prompt(user)
        await increment_usage(user["id"])
        await _increment_global()

        prompt = (
            "Based on this image, give me 5 distinct social post ideas. "
            "Return ONLY a numbered list, one idea per line, max 18 words each. "
            "No preamble, no explanations."
        )
        try:
            response = await _openai_responses_text(
                prompt,
                img["data_b64"],
                img.get("mime", "image/png"),
                "You generate viral, premium social post concepts. Be concrete and creative.",
            )
        except Exception as e:
            await refund_free_prompt(user["id"])
            await _decrement_user_usage(user["id"])
            await _decrement_global()
            log_ai_event(
                user_id=user["id"],
                provider="openai",
                model=OPENAI_TEXT_MODEL,
                prompt_length=len(prompt),
                image_count=1,
                success=False,
                user=user,
            )
            raise HTTPException(status_code=502, detail=f"Idea generation failed: {str(e)[:200]}")

        ideas = []
        for line in (response or "").splitlines():
            s = line.strip()
            if not s:
                continue
            for prefix in ("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "10.", "-", "•"):
                if s.startswith(prefix):
                    s = s[len(prefix):].strip()
                    break
            if s:
                ideas.append(s)
        ideas = ideas[:5] if ideas else [response.strip()]

        if body.session_id:
            await push_message(body.session_id, user["id"], "assistant", "suggestions",
                               content="Here are 5 post ideas you could run with:", suggestions=ideas)
        refreshed = await fs_get("users", user["id"]) or user
        log_ai_event(
            user_id=user["id"],
            provider="openai",
            model=OPENAI_TEXT_MODEL,
            prompt_length=len(prompt),
            image_count=1,
            success=True,
            user=refreshed,
        )
        return {"ideas": ideas, **entitlement_payload(refreshed)}


# ---------- Publish (mock) ----------
@api_router.post("/publish")
async def publish_post(body: PublishBody, user: dict = Depends(get_current_user)):
    img = await fs_get("images", body.image_id)
    if img and img.get("user_id") != user["id"]:
        img = None
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")

    image_url = f"/api/images/{body.image_id}"
    if _backend_base_url := os.environ.get("PUBLIC_BACKEND_BASE_URL", "").rstrip("/"):
        image_url = f"{_backend_base_url}/api/images/{body.image_id}"

    connections = await fs_query("social_accounts", filters=[("user_id", "==", user["id"])], limit=20)
    connections_by_platform = {conn["platform"]: conn for conn in connections if conn.get("status") == "connected"}

    external_url = f"https://{body.platform}.com/mock/{uuid.uuid4().hex[:8]}"
    status_value = "published_mock"
    if body.platform == "facebook" and "facebook" in connections_by_platform:
        fb_conn = connections_by_platform["facebook"]
        meta_post_id = await _publish_facebook_photo(
            page_id=fb_conn["account_id"],
            access_token=fb_conn["access_token"],
            image_url=image_url,
            caption_text=body.caption,
        )
        external_url = f"https://facebook.com/{meta_post_id}"
        status_value = "published_live"
    elif body.platform == "instagram" and "instagram" in connections_by_platform:
        ig_conn = connections_by_platform["instagram"]
        meta_media_id = await _publish_instagram_photo(
            instagram_account_id=ig_conn["account_id"],
            access_token=ig_conn["access_token"],
            image_url=image_url,
            caption_text=body.caption,
        )
        external_url = f"https://instagram.com/p/{meta_media_id}"
        status_value = "published_live"

    pid = str(uuid.uuid4())
    post = {
        "id": pid,
        "user_id": user["id"],
        "image_id": body.image_id,
        "caption": body.caption,
        "platform": body.platform,
        "status": status_value,
        "external_url": external_url,
        "published_at": utcnow().isoformat(),
        "session_id": body.session_id,
    }
    await fs_set("posts", pid, post, merge=False)
    await sync_post_doc(post)

    if body.session_id:
        await push_message(
            body.session_id, user["id"], "assistant", "publish",
            content=f"Posted to {body.platform.title()} (mock).",
            image_id=body.image_id,
            meta={"post_id": pid, "platform": body.platform, "external_url": post["external_url"]},
        )

    return post


@api_router.get("/posts")
async def list_posts(user: dict = Depends(get_current_user)):
    docs = await fs_query("posts", filters=[("user_id", "==", user["id"])], order_by="published_at", descending=True, limit=200)
    return docs


# ---------- Mount router ----------
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("shutdown")
async def shutdown_db_client():
    return None
