import pytest
import asyncio
import json
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi import Response
from fastapi.responses import StreamingResponse

from src.google_api_client import GoogleApiClient, build_gemini_payload_from_openai, build_gemini_payload_from_native


class TestPayloadBuilding:
    """Test payload building functions."""
    
    def test_build_gemini_payload_from_openai(self):
        """Test building Gemini payload from OpenAI format."""
        openai_payload = {
            "model": "gemini-pro",
            "contents": [{"parts": [{"text": "Hello"}]}],
            "systemInstruction": {"parts": [{"text": "You are helpful"}]},
            "generationConfig": {"temperature": 0.7}
        }
        
        result = build_gemini_payload_from_openai(openai_payload)
        
        assert result["model"] == "gemini-pro"
        assert "request" in result
        assert result["request"]["contents"] == openai_payload["contents"]
        assert result["request"]["systemInstruction"] == openai_payload["systemInstruction"]
        assert result["request"]["generationConfig"] == openai_payload["generationConfig"]
        assert "safetySettings" in result["request"]
    
    def test_build_gemini_payload_from_native(self):
        """Test building Gemini payload from native format."""
        native_request = {
            "contents": [{"parts": [{"text": "Hello"}]}],
            "systemInstruction": {"parts": [{"text": "You are helpful"}]}
        }
        model_from_path = "models/gemini-2.5-pro"
        
        result = build_gemini_payload_from_native(native_request, model_from_path)
        
        assert result["model"] == "models/gemini-2.5-pro"
        assert "request" in result
        assert result["request"]["contents"] == native_request["contents"]
        assert "safetySettings" in result["request"]
        assert "generationConfig" in result["request"]
        assert "thinkingConfig" in result["request"]["generationConfig"]
    
    def test_build_gemini_payload_with_search_model(self):
        """Test building payload for search-enabled models."""
        native_request = {
            "contents": [{"parts": [{"text": "What's the weather?"}]}]
        }
        model_from_path = "models/gemini-2.5-pro-search"
        
        result = build_gemini_payload_from_native(native_request, model_from_path)
        
        assert result["model"] == "models/gemini-2.5-pro"  # Base model name
        assert "tools" in result["request"]
        # Should have Google Search tool
        google_search_tools = [tool for tool in result["request"]["tools"] if "googleSearch" in tool]
        assert len(google_search_tools) > 0
    
    def test_build_gemini_payload_with_thinking_variants(self):
        """Test building payload for thinking variants."""
        native_request = {
            "contents": [{"parts": [{"text": "Think about this problem"}]}]
        }
        
        # Test nothinking variant
        result_nothinking = build_gemini_payload_from_native(native_request, "models/gemini-2.5-pro-nothinking")
        thinking_config = result_nothinking["request"]["generationConfig"]["thinkingConfig"]
        assert thinking_config["thinkingBudget"] == 128
        assert thinking_config["includeThoughts"] is True
        
        # Test maxthinking variant
        result_maxthinking = build_gemini_payload_from_native(native_request, "models/gemini-2.5-pro-maxthinking")
        thinking_config = result_maxthinking["request"]["generationConfig"]["thinkingConfig"]
        assert thinking_config["thinkingBudget"] == 32768
        assert thinking_config["includeThoughts"] is True


class TestApiClientIntegration:
    """Integration tests for the API client."""
    
    @pytest.fixture
    def api_client(self):
        return GoogleApiClient()
    
    @pytest.fixture
    def mock_creds(self):
        mock_creds = MagicMock()
        mock_creds.token = "fake-token"
        return mock_creds
    
    def test_invalid_credentials_handling(self, api_client):
        """Test handling of invalid credentials."""
        payload = {"model": "gemini-pro", "request": {"contents": []}}
        
        # Test with None credentials
        response = api_client.send_request(payload, None, "test-project")
        assert response.status_code == 500
        assert "Invalid session" in response.body.decode()
        
        # Test with None project_id
        mock_creds = MagicMock()
        response = api_client.send_request(payload, mock_creds, None)
        assert response.status_code == 500
        assert "Invalid session" in response.body.decode()
    
    @pytest.mark.asyncio
    async def test_streaming_response_format(self, mocker, api_client, mock_creds):
        """Test that streaming responses have correct format."""
        # Mock streaming response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            b'data: {"response": {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}}'
        ]
        
        mock_post = mocker.patch('src.google_api_client.requests.post', return_value=mock_response)
        
        payload = {"model": "gemini-pro", "request": {"contents": []}}
        response = api_client.send_request(payload, mock_creds, "test-project", is_streaming=True)
        
        assert isinstance(response, StreamingResponse)
        assert response.media_type == "text/event-stream"
        
        # Check headers
        assert "Content-Type" in response.headers
        assert response.headers["Content-Type"] == "text/event-stream"
    
    @pytest.mark.asyncio 
    async def test_non_streaming_keepalive_response_format(self, mocker, api_client, mock_creds):
        """Test that non-streaming keepalive responses have correct format."""
        # Mock non-streaming response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"response": {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}}'
        mock_response.encoding = 'utf-8'
        
        mock_post = mocker.patch('src.google_api_client.requests.post', return_value=mock_response)
        
        payload = {"model": "gemini-pro", "request": {"contents": []}}
        
        with patch('src.config.NONSTREAM_KEEPALIVE_ENABLED', True):
            response = api_client.send_request(payload, mock_creds, "test-project", is_streaming=False)
        
        assert isinstance(response, StreamingResponse)
        assert response.media_type == "application/json"
        
        # Consume the response
        content_chunks = []
        async for chunk in response.body_iterator:
            if isinstance(chunk, bytes):
                content_chunks.append(chunk)
            elif isinstance(chunk, str):
                content_chunks.append(chunk.encode('utf-8'))
        
        # Combine chunks and decode
        content = b"".join(content_chunks)
        response_text = content.decode('utf-8')
        assert "Hello" in response_text or "candidates" in response_text
    
    def test_error_response_format(self, mocker, api_client, mock_creds):
        """Test that error responses have correct format."""
        # Mock error response
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": {"message": "Bad request"}}
        
        mock_post = mocker.patch('src.google_api_client.requests.post', return_value=mock_response)
        
        payload = {"model": "gemini-pro", "request": {"contents": []}}
        response = api_client.send_request(payload, mock_creds, "test-project", is_streaming=False)
        
        assert response.status_code == 400
        response_data = json.loads(response.body.decode())
        assert "error" in response_data
        assert response_data["error"]["message"] == "Bad request"


class TestResponseParsing:
    """Test response parsing logic."""
    
    @pytest.fixture
    def api_client(self):
        return GoogleApiClient()
    
    def test_parse_google_api_response_with_data_prefix(self, api_client, mocker):
        """Test parsing responses that start with 'data: '."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = 'data: {"response": {"content": "Hello"}}'
        mock_response.headers = {"Content-Type": "application/json"}
        
        result = api_client._handle_non_streaming_response(mock_response)
        
        assert result.status_code == 200
        response_data = json.loads(result.body.decode())
        assert response_data["content"] == "Hello"
    
    def test_parse_google_api_response_without_data_prefix(self, api_client, mocker):
        """Test parsing responses without 'data: ' prefix."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"response": {"content": "Hello"}}'
        mock_response.headers = {"Content-Type": "application/json"}
        
        result = api_client._handle_non_streaming_response(mock_response)
        
        assert result.status_code == 200
        response_data = json.loads(result.body.decode())
        assert response_data["content"] == "Hello"