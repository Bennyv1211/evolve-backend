import os
import io
import uuid
import pytest
import requests
from PIL import Image, ImageDraw

BASE_URL = os.environ.get("REACT_APP_BACKEND_URL", "https://chat-content-studio.preview.emergentagent.com").rstrip("/")
API = f"{BASE_URL}/api"

DEMO_EMAIL = "demo@evolve.ai"
DEMO_PASSWORD = "demo1234"


def _build_real_jpeg() -> bytes:
    """Build a real JPEG with visual features (objects, edges, textures)."""
    img = Image.new("RGB", (512, 512), (240, 230, 210))
    d = ImageDraw.Draw(img)
    # gradient-ish bands
    for y in range(0, 512, 8):
        d.rectangle([0, y, 512, y + 4], fill=(200 - y // 4, 120 + y // 6, 80 + y // 8))
    # shapes
    d.ellipse([60, 60, 260, 260], fill=(220, 60, 80), outline=(40, 40, 40), width=4)
    d.rectangle([280, 140, 470, 380], fill=(70, 130, 200), outline=(20, 20, 20), width=3)
    d.polygon([(120, 320), (260, 470), (40, 470)], fill=(80, 180, 90), outline=(10, 10, 10))
    # texture lines
    for i in range(0, 512, 16):
        d.line([(0, i), (512, i)], fill=(0, 0, 0, 80), width=1)
    d.text((30, 20), "Evolve Test", fill=(20, 20, 20))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


REAL_JPEG_BYTES = _build_real_jpeg()


@pytest.fixture(scope="session")
def api_base():
    return API


@pytest.fixture(scope="session")
def session_client():
    s = requests.Session()
    s.headers.update({"Accept": "application/json"})
    return s


@pytest.fixture(scope="session")
def demo_token(session_client):
    r = session_client.post(f"{API}/auth/login", json={"email": DEMO_EMAIL, "password": DEMO_PASSWORD}, timeout=30)
    if r.status_code != 200:
        # try register if not exists
        rr = session_client.post(f"{API}/auth/register",
                                 json={"email": DEMO_EMAIL, "password": DEMO_PASSWORD, "name": "Demo User"}, timeout=30)
        if rr.status_code == 200:
            return rr.json()["token"]
        pytest.skip(f"Cannot authenticate demo user. login={r.status_code} register={rr.status_code}")
    return r.json()["token"]


@pytest.fixture(scope="session")
def demo_headers(demo_token):
    return {"Authorization": f"Bearer {demo_token}"}


@pytest.fixture(scope="session")
def fresh_user(session_client):
    """Register a brand-new throwaway user for isolation-sensitive tests."""
    email = f"test_{uuid.uuid4().hex[:10]}@example.com"
    pwd = "Passw0rd!"
    r = session_client.post(f"{API}/auth/register",
                            json={"email": email, "password": pwd, "name": "Fresh Test"}, timeout=30)
    assert r.status_code == 200, f"register failed: {r.status_code} {r.text}"
    data = r.json()
    return {"email": email, "password": pwd, "token": data["token"], "user": data["user"]}


@pytest.fixture(scope="session")
def fresh_headers(fresh_user):
    return {"Authorization": f"Bearer {fresh_user['token']}"}


@pytest.fixture
def real_jpeg_bytes():
    return REAL_JPEG_BYTES
