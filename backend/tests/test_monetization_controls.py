"""Integration tests for free-prompt paywall enforcement.

These tests avoid unnecessary AI spend by manipulating Mongo state directly
where possible and asserting the API blocks or allows requests correctly.
"""

import asyncio
import os
import uuid

import requests
from motor.motor_asyncio import AsyncIOMotorClient

from .conftest import API


async def _set_user_fields(user_id: str, values: dict):
    mongo_url = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
    db_name = os.environ.get("DB_NAME", "evolve_creator_db")
    client = AsyncIOMotorClient(mongo_url)
    try:
        await client[db_name].users.update_one({"id": user_id}, {"$set": values})
    finally:
        client.close()


def _register(session: requests.Session):
    email = f"monetize_{uuid.uuid4().hex[:8]}@example.com"
    password = "Passw0rd!"
    response = session.post(
        f"{API}/auth/register",
        json={"email": email, "password": password, "name": "Monetization Test"},
        timeout=30,
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_fourth_prompt_hits_paywall(session_client, real_jpeg_bytes):
    data = _register(session_client)
    headers = {"Authorization": f"Bearer {data['token']}"}
    user_id = data["user"]["id"]

    files = {"file": ("gating.jpg", real_jpeg_bytes, "image/jpeg")}
    upload = session_client.post(f"{API}/images/upload", headers=headers, files=files, timeout=60)
    assert upload.status_code == 200, upload.text
    image_id = upload.json()["id"]

    session = session_client.post(
        f"{API}/chat/sessions",
        headers=headers,
        json={"title": "paywall"},
        timeout=30,
    )
    assert session.status_code == 200
    session_id = session.json()["id"]

    asyncio.run(_set_user_fields(user_id, {"free_prompts_used": 3, "subscription_status": "free"}))

    response = session_client.post(
        f"{API}/chat/edit-image",
        headers=headers,
        json={
            "session_id": session_id,
            "image_id": image_id,
            "prompt": "Make this image premium",
        },
        timeout=30,
    )
    assert response.status_code == 402, response.text
    detail = response.json()["detail"]
    assert detail["paywall_required"] is True
    assert detail["free_prompts_remaining"] == 0


def test_subscribed_user_not_blocked_by_free_prompt_limit(session_client):
    data = _register(session_client)
    headers = {"Authorization": f"Bearer {data['token']}"}
    user_id = data["user"]["id"]

    session = session_client.post(
        f"{API}/chat/sessions",
        headers=headers,
        json={"title": "subscribed"},
        timeout=30,
    )
    assert session.status_code == 200
    session_id = session.json()["id"]

    asyncio.run(
        _set_user_fields(
            user_id,
            {
                "free_prompts_used": 3,
                "subscription_status": "active",
                "subscription_expires_at": None,
            },
        )
    )

    bogus_image_id = str(uuid.uuid4())
    response = session_client.post(
        f"{API}/chat/edit-image",
        headers=headers,
        json={
            "session_id": session_id,
            "image_id": bogus_image_id,
            "prompt": "Try to continue",
        },
        timeout=30,
    )
    # The request should get past the paywall gate and then fail at image lookup.
    assert response.status_code != 402, response.text
    assert response.status_code in (404, 429, 502)
