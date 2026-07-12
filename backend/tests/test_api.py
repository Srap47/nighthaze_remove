"""API endpoint tests using the FastAPI TestClient."""

import pytest


def test_health_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200

    data = response.json()
    assert "model_loaded" in data
    assert "version" in data
    assert data["status"] == "ok"


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "NightHaze API"


@pytest.mark.slow
def test_upload_valid_jpeg(client, fixture_path):
    with open(fixture_path, "rb") as handle:
        response = client.post(
            "/api/v1/dehaze/upload",
            files={"image": ("test.jpg", handle, "image/jpeg")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["dehazed_image_b64"].startswith("data:image/png;base64,")
    assert len(data["pipeline_stages"]) == 6


def test_upload_invalid_file_type(client):
    response = client.post(
        "/api/v1/dehaze/upload",
        files={"image": ("test.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["success"] is False
    assert body["error"] == "bad_request"
    assert body["detail"]


def test_upload_oversized_file(client):
    large_bytes = b"0" * (11 * 1024 * 1024)  # 11MB > 10MB limit
    response = client.post(
        "/api/v1/dehaze/upload",
        files={"image": ("big.jpg", large_bytes, "image/jpeg")},
    )

    assert response.status_code == 413
    body = response.json()
    assert body["success"] is False
    assert body["error"] == "payload_too_large"


def test_upload_missing_file_is_validation_error(client):
    response = client.post("/api/v1/dehaze/upload")

    assert response.status_code == 422
    body = response.json()
    assert body["success"] is False
    assert body["error"] == "validation_error"


@pytest.mark.slow
def test_demo_endpoint(client):
    response = client.get("/api/v1/dehaze/demo")

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["original_image_b64"].startswith("data:image/png;base64,")


def test_unknown_route_returns_uniform_envelope(client):
    response = client.get("/api/v1/does-not-exist")

    assert response.status_code == 404
    assert response.json() == {
        "success": False,
        "error": "not_found",
        "detail": "Not Found",
    }
