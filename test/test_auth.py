import pytest
from fastapi import FastAPI, Depends, Response, HTTPException
from httpx import AsyncClient, ASGITransport
from unittest.mock import MagicMock, patch

# We need to import the module to be tested
from src import auth

# Define a dummy protected endpoint using the dependency
app = FastAPI()

@app.get("/protected")
async def protected_route(session: tuple = Depends(auth.get_current_session)):
    # The dependency itself handles success/failure. If it succeeds,
    # we just return a success message. If it fails, it raises an HTTPException.
    return {"status": "ok", "project_id": session[1]}

# --- Test Cases ---

@pytest.mark.asyncio
async def test_retry_on_failure_then_success(mocker):
    """
    Tests that the get_current_session dependency successfully retries
    after an initial failure.
    """
    # Mock the underlying functions called by get_current_session
    mock_creds = MagicMock()
    mock_project_id = "test-project"
    mock_file_path = "/fake/path/creds.json"

    # First call fails (no creds), second call succeeds
    mock_get_creds = mocker.patch(
        'src.auth.get_credentials',
        side_effect=[
            (None, None, None),
            (mock_creds, mock_project_id, mock_file_path)
        ]
    )
    # Mock onboard_user to do nothing and return None
    mock_onboard = mocker.patch('src.auth.onboard_user', return_value=None)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/protected")

    # Assertions
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "project_id": mock_project_id}
    assert mock_get_creds.call_count == 2
    mock_onboard.assert_called_once_with(mock_creds, mock_project_id, mock_file_path)

@pytest.mark.asyncio
async def test_retry_exhausted_fails(mocker):
    """
    Tests that the get_current_session dependency fails with a 503 error
    after all retry attempts are exhausted.
    """
    # Mock get_credentials to consistently fail by returning None
    mock_get_creds = mocker.patch('src.auth.get_credentials', return_value=(None, None, None))
    mock_onboard = mocker.patch('src.auth.onboard_user') # Not expected to be called

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/protected")

    # Assertions
    assert response.status_code == 503
    assert "No available credentials" in response.json()["detail"]
    assert mock_get_creds.call_count == 3
    mock_onboard.assert_not_called()

@pytest.mark.asyncio
async def test_onboard_user_failure_retries(mocker):
    """
    Tests that the get_current_session dependency retries if onboarding fails.
    """
    # Mock get_credentials to always return a valid credential
    mock_creds1, mock_creds2, mock_creds3 = MagicMock(), MagicMock(), MagicMock()
    mock_get_creds = mocker.patch(
        'src.auth.get_credentials',
        side_effect=[
            (mock_creds1, "proj1", "path1"),
            (mock_creds2, "proj2", "path2"),
            (mock_creds3, "proj3", "path3"),
        ]
    )
    # Mock onboard_user to fail on the first two attempts, then succeed
    mock_onboard = mocker.patch(
        'src.auth.onboard_user',
        side_effect=[Exception("Onboarding failed"), Exception("Onboarding failed again"), None]
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/protected")

    # Assertions
    assert response.status_code == 200
    assert response.json()["project_id"] == "proj3"
    assert mock_get_creds.call_count == 3
    assert mock_onboard.call_count == 3