import pytest
import requests
from unittest.mock import MagicMock, patch
from fastapi import Response

from src.google_api_client import GoogleApiClient

@pytest.fixture
def api_client():
    """Provides a GoogleApiClient instance for testing."""
    return GoogleApiClient()

def test_send_request_retries_on_api_failure(mocker, api_client):
    """
    Tests that send_request retries the API call using the retry_api_call decorator
    when a RequestException occurs.
    """
    # Mock credentials and payload
    mock_creds = MagicMock()
    mock_creds.token = "fake-token"
    mock_project_id = "test-project"
    payload = {
        "model": "gemini-pro",
        "request": {"contents": [{"parts": [{"text": "Hello"}]}]}
    }

    # Mock the response from requests.post
    mock_successful_response = MagicMock(spec=requests.Response)
    mock_successful_response.status_code = 200
    mock_successful_response.text = 'data: {"response": {"content": "Hi there!"}}' 
    mock_successful_response.json.return_value = {"response": {"content": "Hi there!"}}
    mock_successful_response.headers = {"Content-Type": "application/json"}

    # Setup the side effect for requests.post: fail twice, succeed on the third attempt
    mock_post = mocker.patch(
        'src.google_api_client.requests.post',
        side_effect=[
            requests.exceptions.RequestException("Connection error"),
            requests.exceptions.RequestException("Timeout"),
            mock_successful_response
        ]
    )

    # Call the method to be tested
    response = api_client.send_request(payload, mock_creds, mock_project_id, is_streaming=False)

    # Assertions
    assert mock_post.call_count == 3
    assert isinstance(response, Response)
    assert response.status_code == 200
    response_content = bytes(response.body).decode()
    assert 'Hi there!' in response_content

def test_send_request_fails_after_exhausting_retries(mocker, api_client):
    """
    Tests that send_request raises an exception after all retries are exhausted.
    """
    # Mock credentials and payload
    mock_creds = MagicMock()
    mock_creds.token = "fake-token"
    mock_project_id = "test-project"
    payload = {
        "model": "gemini-pro",
        "request": {"contents": [{"parts": [{"text": "Hello"}]}]}
    }

    # Setup requests.post to always fail
    mock_post = mocker.patch(
        'src.google_api_client.requests.post',
        side_effect=requests.exceptions.RequestException("Persistent connection error")
    )

    # Call the method and assert that it raises the final exception
    # The exception is caught inside send_request and converted to a 500 response
    response = api_client.send_request(payload, mock_creds, mock_project_id, is_streaming=False)

    # Assertions
    assert mock_post.call_count == 3
    assert response.status_code == 500
    response_content = bytes(response.body).decode()
    assert "Request failed after retries" in response_content