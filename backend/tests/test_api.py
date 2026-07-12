"""API endpoint tests using the FastAPI TestClient.

Tests verify:
- Health/readiness endpoint (model status, version)
- Root documentation endpoint
- Upload endpoint (JPEG/PNG/WebP, size limits, error handling)
- Demo endpoint (bundled sample image)
- Error response format consistency (uniform ErrorResponse envelope)
"""

import pytest


def test_health_endpoint(client):
    """Verify GET /health returns service status and deployment info.

    Used by Kubernetes, load balancers, and frontend to check readiness.
    Includes model_loaded flag (critical for knowing if dehazing is available),
    device (GPU/CPU), and version for deployment tracking.
    """
    response = client.get("/api/v1/health")
    assert response.status_code == 200

    data = response.json()
    assert "model_loaded" in data  # Critical: model available?
    assert "version" in data       # Deployment tracking
    assert data["status"] == "ok"


def test_root_endpoint(client):
    """Verify GET / returns welcome message and docs link.

    Simple endpoint to verify API is reachable and responsive.
    Points to /docs for OpenAPI documentation.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "NightHaze API"


@pytest.mark.slow
def test_upload_valid_jpeg(client, fixture_path):
    """Verify successful POST /dehaze/upload with valid JPEG image.

    Uploads a real image file, checks:
    - HTTP 200 (success)
    - success=true in response
    - dehazed_image_b64 present and valid (PNG data URI)
    - all 6 pipeline stages completed and reported

    Marked slow because it triggers FFA-Net inference (~10-20s).
    """
    with open(fixture_path, "rb") as handle:
        response = client.post(
            "/api/v1/dehaze/upload",
            files={"image": ("test.jpg", handle, "image/jpeg")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["dehazed_image_b64"].startswith("data:image/png;base64,")
    assert len(data["pipeline_stages"]) == 6  # All 6 stages executed


def test_upload_invalid_file_type(client):
    """Verify POST /dehaze/upload rejects unsupported MIME types.

    Attempts to upload a text file (MIME type text/plain).
    Should be rejected before processing with HTTP 400 (bad request).
    Checks error response format and error code.
    """
    response = client.post(
        "/api/v1/dehaze/upload",
        files={"image": ("test.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 400  # Bad request
    body = response.json()
    assert body["success"] is False
    assert body["error"] == "bad_request"  # Consistent error code
    assert body["detail"]  # Has descriptive message


def test_upload_oversized_file(client):
    """Verify POST /dehaze/upload enforces file size limit.

    Attempts to upload an 11MB file (exceeds 10MB limit).
    Should be rejected with HTTP 413 (Payload Too Large).
    Tests the size check before processing (fail-fast).
    """
    large_bytes = b"0" * (11 * 1024 * 1024)  # 11MB > 10MB limit
    response = client.post(
        "/api/v1/dehaze/upload",
        files={"image": ("big.jpg", large_bytes, "image/jpeg")},
    )

    assert response.status_code == 413  # Payload Too Large
    body = response.json()
    assert body["success"] is False
    assert body["error"] == "payload_too_large"


def test_upload_missing_file_is_validation_error(client):
    """Verify POST /dehaze/upload requires the image file parameter.

    Attempts to upload without the 'image' file parameter.
    Pydantic validation should catch this and return HTTP 422 (Unprocessable Entity).
    """
    response = client.post("/api/v1/dehaze/upload")

    assert response.status_code == 422  # Validation error
    body = response.json()
    assert body["success"] is False
    assert body["error"] == "validation_error"


@pytest.mark.slow
def test_demo_endpoint(client):
    """Verify GET /dehaze/demo processes bundled sample image without upload.

    Allows frontend to demonstrate dehazing capability without user upload.
    Uses pre-bundled sample nighttime photo (sample_hazy_night.jpg).
    Checks:
    - HTTP 200 (success)
    - success=true
    - original_image_b64 present (same as dehazed for comparison)

    Marked slow because it triggers FFA-Net inference (~10-20s).
    """
    response = client.get("/api/v1/dehaze/demo")

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["original_image_b64"].startswith("data:image/png;base64,")


def test_unknown_route_returns_uniform_envelope(client):
    """Verify 404 (unknown route) returns consistent ErrorResponse format.

    Tests that the exception handler for Starlette HTTPException
    properly wraps 404s in the standard error envelope:
    { success: false, error: "not_found", detail: "..." }
    Ensures all error responses (not just domain errors) follow the same format.
    """
    response = client.get("/api/v1/does-not-exist")

    assert response.status_code == 404
    assert response.json() == {
        "success": False,
        "error": "not_found",
        "detail": "Not Found",
    }
