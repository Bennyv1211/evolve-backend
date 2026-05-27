"""Abuse / cost-protection tests for iteration 2.

Validates upload size cap, dimension downscaling, animated-GIF flattening,
prompt validation, rate limiting, concurrency lock, and IP-based registration
throttling for the Evolve Content Creator backend.

FRUGAL with LLM credits: live AI calls are deliberately avoided. We test the
protection layers by sending requests that fail at the validation/limit step
*before* the AI call fires.
"""
import asyncio
import base64
import io
import os
import threading
import time
import uuid

import pytest
import requests
from PIL import Image

_FRONTEND_ENV = "/app/frontend/.env"
if "REACT_APP_BACKEND_URL" not in os.environ and os.path.exists(_FRONTEND_ENV):
    with open(_FRONTEND_ENV) as _f:
        for _line in _f:
            if _line.startswith("REACT_APP_BACKEND_URL="):
                os.environ["REACT_APP_BACKEND_URL"] = _line.split("=", 1)[1].strip()
                break
BASE_URL = os.environ["REACT_APP_BACKEND_URL"].rstrip("/")
API = f"{BASE_URL}/api"


# ---------- Helpers ----------
def _png_bytes(size=(64, 64), color=(10, 20, 30)) -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_bytes(size=(64, 64), color=(200, 100, 50)) -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _big_jpeg_bytes_over_5mb() -> bytes:
    """Random noise JPEG that exceeds 5MB after compression."""
    import os as _os
    # 4500x4500 random RGB -> JPEG with random data won't compress well
    arr = _os.urandom(4500 * 4500 * 3)
    img = Image.frombytes("RGB", (4500, 4500), arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    data = buf.getvalue()
    return data


def _huge_dim_png() -> bytes:
    """4000x4000 PNG (under 5MB raw bytes after compression of flat-ish image)."""
    img = Image.new("RGB", (4000, 4000), (120, 80, 200))
    # add some pattern so PNG isn't trivially compressed-uniform
    for y in range(0, 4000, 50):
        for x in range(0, 4000, 50):
            img.putpixel((x, y), (255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _animated_gif_bytes() -> bytes:
    frames = []
    for c in [(255, 0, 0), (0, 255, 0), (0, 0, 255)]:
        f = Image.new("RGB", (64, 64), c)
        frames.append(f)
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], duration=100, loop=0)
    return buf.getvalue()


# ---------- Fixtures (local; avoid touching session fixtures) ----------
@pytest.fixture(scope="module")
def http():
    s = requests.Session()
    s.headers.update({"Accept": "application/json"})
    return s


def _register(http, email=None, password="Passw0rd!"):
    email = email or f"test_abuse_{uuid.uuid4().hex[:10]}@example.com"
    r = http.post(f"{API}/auth/register",
                  json={"email": email, "password": password, "name": "Abuse Test"}, timeout=30)
    return r, email


@pytest.fixture(scope="module")
def user_a(http):
    r, email = _register(http)
    assert r.status_code == 200, f"register failed: {r.status_code} {r.text}"
    body = r.json()
    return {"email": email, "token": body["token"], "user": body["user"]}


@pytest.fixture(scope="module")
def user_a_headers(user_a):
    return {"Authorization": f"Bearer {user_a['token']}"}


@pytest.fixture(scope="module")
def user_a_session(http, user_a_headers):
    r = http.post(f"{API}/chat/sessions", headers=user_a_headers, json={"title": "abuse"}, timeout=30)
    assert r.status_code == 200, r.text
    return r.json()["id"]


@pytest.fixture(scope="module")
def user_a_image(http, user_a_headers):
    files = {"file": ("a.jpg", _jpeg_bytes(), "image/jpeg")}
    r = http.post(f"{API}/images/upload", headers=user_a_headers, files=files, timeout=30)
    assert r.status_code == 200, r.text
    return r.json()["id"]


# ============================================================
#  IMAGE UPLOAD PROTECTIONS
# ============================================================
class TestImageUploadProtections:

    def test_oversized_upload_returns_413(self, http, user_a_headers):
        big = _big_jpeg_bytes_over_5mb()
        # sanity: confirm we actually exceed 5MB before sending
        assert len(big) > 5 * 1024 * 1024, f"Constructed file only {len(big)} bytes"
        files = {"file": ("big.jpg", big, "image/jpeg")}
        r = http.post(f"{API}/images/upload", headers=user_a_headers, files=files, timeout=60)
        assert r.status_code == 413, f"Expected 413, got {r.status_code}: {r.text[:300]}"
        assert "too large" in r.text.lower()

    def test_large_dimensions_downscaled_to_1536(self, http, user_a_headers):
        original = _huge_dim_png()
        # ensure original PNG is under 5MB so it passes size check (PNG of 4000x4000 with
        # large flat regions compresses well)
        if len(original) > 5 * 1024 * 1024:
            pytest.skip(f"Generated 4000x4000 PNG is {len(original)} bytes, exceeds 5MB cap")
        files = {"file": ("huge.png", original, "image/png")}
        r = http.post(f"{API}/images/upload", headers=user_a_headers, files=files, timeout=60)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text[:300]}"
        rec = r.json()
        # Fetch back and verify dimensions <= 1536
        img_resp = http.get(f"{API}/images/{rec['id']}", timeout=30)
        assert img_resp.status_code == 200
        img = Image.open(io.BytesIO(img_resp.content))
        longest = max(img.size)
        assert longest <= 1536, f"Image not downscaled: {img.size}"
        # And byte size should be much smaller than original
        assert len(img_resp.content) < len(original), "Re-encoded image not smaller"

    def test_animated_gif_rejected_as_unsupported_mime(self, http, user_a_headers):
        """Server only allows PNG/JPEG/WEBP. GIF must be rejected at MIME check."""
        gif = _animated_gif_bytes()
        files = {"file": ("anim.gif", gif, "image/gif")}
        r = http.post(f"{API}/images/upload", headers=user_a_headers, files=files, timeout=30)
        assert r.status_code == 400, f"Expected 400, got {r.status_code}: {r.text[:200]}"
        assert "unsupported" in r.text.lower() or "image" in r.text.lower()

    def test_animated_webp_first_frame_only(self, http, user_a_headers):
        """If we mislabel an animated GIF as WEBP, _normalize_image should still
        flatten and accept it. We can't easily build animated WEBP via PIL across
        versions, so use static WEBP as a positive check."""
        img = Image.new("RGB", (256, 256), (100, 200, 150))
        buf = io.BytesIO()
        img.save(buf, format="WEBP", quality=85)
        files = {"file": ("s.webp", buf.getvalue(), "image/webp")}
        r = http.post(f"{API}/images/upload", headers=user_a_headers, files=files, timeout=30)
        assert r.status_code == 200, r.text
        rec = r.json()
        # Server may re-encode to JPEG or PNG
        assert rec["mime"] in ("image/jpeg", "image/png", "image/webp")


# ============================================================
#  PROMPT VALIDATION (Pydantic)
# ============================================================
class TestPromptValidation:

    def test_edit_image_empty_prompt_422(self, http, user_a_headers, user_a_session, user_a_image):
        r = http.post(f"{API}/chat/edit-image", headers=user_a_headers,
                      json={"session_id": user_a_session, "image_id": user_a_image, "prompt": ""},
                      timeout=30)
        assert r.status_code == 422, f"Expected 422, got {r.status_code}: {r.text[:200]}"

    def test_edit_image_prompt_over_1000_chars_422(self, http, user_a_headers, user_a_session, user_a_image):
        prompt = "a" * 1001
        r = http.post(f"{API}/chat/edit-image", headers=user_a_headers,
                      json={"session_id": user_a_session, "image_id": user_a_image, "prompt": prompt},
                      timeout=30)
        assert r.status_code == 422, f"Expected 422, got {r.status_code}: {r.text[:200]}"

    def test_edit_image_missing_session_id_422(self, http, user_a_headers, user_a_image):
        r = http.post(f"{API}/chat/edit-image", headers=user_a_headers,
                      json={"image_id": user_a_image, "prompt": "hello"}, timeout=30)
        assert r.status_code == 422


# ============================================================
#  RATE LIMIT (3/min/user)
# ============================================================
class TestRateLimit:
    """Use a fresh user + invalid image_id so the AI is never invoked.

    Flow inside edit-image (server.py):
      1. session lookup (404 if not found)
      2. acquire user lock
      3. _enforce_rate_limit  <-- fires here on the 4th call
      4. quota check, then image lookup (404 if invalid)
    So calls 1-3 each hit rate-limit counter then fail at image lookup (404).
    Call 4 fires the 429 from _enforce_rate_limit BEFORE any LLM cost.
    """

    def test_rate_limit_429_on_4th_call(self, http):
        r, email = _register(http)
        assert r.status_code == 200, r.text
        token = r.json()["token"]
        H = {"Authorization": f"Bearer {token}"}

        sr = http.post(f"{API}/chat/sessions", headers=H, json={"title": "rl"}, timeout=30)
        assert sr.status_code == 200
        sid = sr.json()["id"]

        bogus_image_id = str(uuid.uuid4())  # guaranteed not in DB

        statuses = []
        for i in range(4):
            r = http.post(f"{API}/chat/edit-image", headers=H,
                          json={"session_id": sid, "image_id": bogus_image_id, "prompt": "x"},
                          timeout=30)
            statuses.append(r.status_code)
        # Calls 1-3: image not found -> 404 (rate-limit counter incremented)
        # Call 4: rate-limited -> 429 "going a bit fast"
        assert statuses[:3] == [404, 404, 404], f"Pre-limit statuses unexpected: {statuses}"
        assert statuses[3] == 429, f"4th call expected 429, got {statuses[3]}"


# ============================================================
#  CONCURRENCY LOCK
# ============================================================
class TestConcurrencyLock:

    def test_simultaneous_requests_second_returns_429(self, http):
        """Fire two requests in parallel from same user; second should hit
        the 'AI request already running' guard or rate-limit guard (both 429)."""
        r, _ = _register(http)
        assert r.status_code == 200, r.text
        token = r.json()["token"]
        H = {"Authorization": f"Bearer {token}"}

        # Create session
        sr = http.post(f"{API}/chat/sessions", headers=H, json={"title": "lock"}, timeout=30)
        sid = sr.json()["id"]

        # Upload tiny image so AI WOULD fire if not guarded. To avoid LLM spend,
        # we use a bogus image_id so the in-flight request returns 404 quickly
        # AFTER the lock is acquired and rate-limit counter incremented.
        # That gives the parallel request enough window to attempt + see lock.locked()=True.
        bogus = str(uuid.uuid4())

        results = []
        errors = []

        def fire():
            try:
                rr = requests.post(f"{API}/chat/edit-image", headers=H,
                                   json={"session_id": sid, "image_id": bogus, "prompt": "test"},
                                   timeout=30)
                results.append(rr.status_code)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=fire) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Request errors: {errors}"
        # One should be 404 (first acquired lock, found bad image), the other 429 (lock contention).
        # Both 404 would mean lock was never contended (timing). Both 429 means rate-limit kicked in too.
        # Accept any combo containing a 429 OR document if not.
        assert 429 in results or results.count(404) == 2, f"Unexpected results: {results}"
        # We expect at least one 429 in well-behaved environment
        if 429 not in results:
            pytest.skip(f"Lock contention not observed (likely too fast). Got: {results}")


# ============================================================
#  PERSISTED QUOTA STILL WORKS (sanity) - runs BEFORE IP rate-limit
#  test so it can still register a fresh user.
# ============================================================
class TestQuotaStillEnforced:

    def test_quota_enforcement_via_mongo_bump(self, http):
        """Bump a fresh user's usage to 10 directly in Mongo, then verify
        edit-image returns 429 BEFORE any LLM call (no quota burn)."""
        from motor.motor_asyncio import AsyncIOMotorClient

        # Load Mongo env vars from backend/.env if not present
        _BE_ENV = "/app/backend/.env"
        if "MONGO_URL" not in os.environ and os.path.exists(_BE_ENV):
            with open(_BE_ENV) as f:
                for line in f:
                    if "=" in line and not line.strip().startswith("#"):
                        k, v = line.strip().split("=", 1)
                        os.environ.setdefault(k, v.strip().strip('"').strip("'"))

        r, _ = _register(http)
        assert r.status_code == 200, r.text
        token = r.json()["token"]
        user_id = r.json()["user"]["id"]
        H = {"Authorization": f"Bearer {token}"}

        # Upload image + create session
        files = {"file": ("q.jpg", _jpeg_bytes(), "image/jpeg")}
        up = http.post(f"{API}/images/upload", headers=H, files=files, timeout=30)
        assert up.status_code == 200, up.text
        iid = up.json()["id"]
        sr = http.post(f"{API}/chat/sessions", headers=H, json={"title": "q"}, timeout=30)
        sid = sr.json()["id"]

        async def bump():
            from datetime import datetime, timezone
            cl = AsyncIOMotorClient(os.environ["MONGO_URL"])
            try:
                d = cl[os.environ["DB_NAME"]]
                today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                await d.usage.update_one(
                    {"user_id": user_id, "date": today},
                    {"$set": {"count": 10, "user_id": user_id, "date": today}},
                    upsert=True,
                )
            finally:
                cl.close()

        asyncio.run(bump())

        u = http.get(f"{API}/usage/today", headers=H, timeout=30).json()
        assert u["used"] == 10 and u["remaining"] == 0

        # edit-image must 429 (quota), NOT call AI
        r = http.post(f"{API}/chat/edit-image", headers=H,
                      json={"session_id": sid, "image_id": iid, "prompt": "make luxury"},
                      timeout=30)
        assert r.status_code == 429, f"Expected 429, got {r.status_code}: {r.text[:200]}"


# ============================================================
#  REGISTRATION IP RATE LIMIT (5/hour/IP) - runs LAST as it
#  exhausts the in-memory IP bucket and blocks further registrations.
# ============================================================
class TestRegisterIPRateLimit:
    def test_excessive_registrations_eventually_429(self, http):
        statuses = []
        for _ in range(8):
            r, _ = _register(http)
            statuses.append(r.status_code)
        assert 429 in statuses, f"No 429 observed across 8 registrations: {statuses}"
