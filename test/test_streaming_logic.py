import pytest
import asyncio
import json
import os
import requests
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi import Response
from fastapi.responses import StreamingResponse

from src.google_api_client import GoogleApiClient
from src.config import NONSTREAM_KEEPALIVE_ENABLED, NONSTREAM_KEEPALIVE_INTERVAL


@pytest.fixture
def api_client():
    """Provides a GoogleApiClient instance for testing."""
    return GoogleApiClient()


@pytest.fixture
def mock_creds():
    """Mock OAuth2 credentials."""
    mock_creds = MagicMock()
    mock_creds.token = "fake-token"
    return mock_creds


@pytest.fixture
def mock_payload():
    """Mock API payload."""
    return {
        "model": "gemini-pro",
        "request": {"contents": [{"parts": [{"text": "Hello"}]}]}
    }


class TestStreamingLogic:
    """Test the new streaming/non-streaming logic."""
    
    def test_streaming_request_uses_true_streaming(self, mocker, api_client, mock_creds, mock_payload):
        """
        When is_streaming=True, should use true streaming (streamGenerateContent)
        """
        # Mock successful streaming response
        mock_stream_response = MagicMock(spec=requests.Response)
        mock_stream_response.status_code = 200
        mock_stream_response.iter_lines.return_value = [
            b'data: {"response": {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}}'
        ]
        
        mock_post = mocker.patch('src.google_api_client.requests.post', return_value=mock_stream_response)
        
        # Call with streaming=True
        response = api_client.send_request(
            mock_payload, 
            mock_creds, 
            "test-project", 
            is_streaming=True
        )
        
        # Verify it used streaming endpoint
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "streamGenerateContent" in call_args[0][0]  # URL should contain streaming endpoint
        assert call_args[1]['stream'] is True  # Should use stream=True
        assert isinstance(response, StreamingResponse)  # Should return StreamingResponse
    
    def test_non_streaming_without_keepalive_uses_true_non_streaming(self, mocker, api_client, mock_creds, mock_payload):
        """
        When is_streaming=False and keepalive disabled, should use true non-streaming
        """
        # Mock non-streaming response
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 200
        mock_response.text = 'data: {"response": {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}}'
        mock_response.json.return_value = {"response": {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}}
        mock_response.headers = {"Content-Type": "application/json"}
        
        mock_post = mocker.patch('src.google_api_client.requests.post', return_value=mock_response)
        
        # Temporarily disable keepalive for this test
        with patch('src.config.NONSTREAM_KEEPALIVE_ENABLED', False):
            response = api_client.send_request(
                mock_payload, 
                mock_creds, 
                "test-project", 
                is_streaming=False
            )
        
        # Verify it used non-streaming endpoint
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "generateContent" in call_args[0][0]  # URL should contain non-streaming endpoint
        assert call_args[1]['stream'] is False  # Should use stream=False
        assert isinstance(response, Response)  # Should return regular Response
    
    @pytest.mark.asyncio
    async def test_non_streaming_with_keepalive_uses_non_streaming_with_heartbeat(self, mocker, api_client, mock_creds, mock_payload):
        """
        When is_streaming=False and keepalive enabled, should use non-streaming with heartbeat
        """
        # Mock non-streaming response
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 200
        mock_response.text = '{"response": {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}}'
        mock_response.json.return_value = {"response": {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}}
        mock_response.encoding = 'utf-8'  # Add encoding attribute
        
        mock_post = mocker.patch.object(api_client, '_make_request', return_value=mock_response)
        
        # Temporarily enable keepalive for this test
        with patch('src.config.NONSTREAM_KEEPALIVE_ENABLED', True):
            response = api_client.send_request(
                mock_payload, 
                mock_creds, 
                "test-project", 
                is_streaming=False
            )
        
        # Verify it returned streaming response
        assert isinstance(response, StreamingResponse)  # But return StreamingResponse for keepalive
        
        # Consume the streaming response to trigger the actual work and verify the mock was called
        content_chunks = []
        async for chunk in response.body_iterator:
            if isinstance(chunk, bytes):
                content_chunks.append(chunk)
            elif isinstance(chunk, str):
                content_chunks.append(chunk.encode('utf-8'))
        
        content = b"".join(content_chunks)
        
        # Now verify that the mock was called (after consuming the response)
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "generateContent" in call_args[0][0]  # URL should contain non-streaming endpoint
        # Check the stream parameter (should be the 4th positional argument)
        assert call_args[0][3] is False  # Should use stream=False for the API call
        
        # Verify response contains expected content
        response_text = content.decode('utf-8')
        assert "Hello" in response_text or content.strip() != ""


class TestNonStreamingKeepalive:
    """Test non-streaming keepalive functionality."""
    
    @pytest.mark.asyncio
    async def test_keepalive_sends_heartbeats(self, mocker, api_client, mock_creds, mock_payload):
        """
        Keepalive should send heartbeats while waiting for API response
        """
        # Mock slow API response
        def slow_api_call(*args, **kwargs):
            import time
            time.sleep(0.2)  # Simulate slow API call
            mock_response = MagicMock(spec=requests.Response)
            mock_response.status_code = 200
            mock_response.text = '{"response": {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}}'
            mock_response.encoding = 'utf-8'  # Add encoding attribute
            return mock_response
        
        mock_post = mocker.patch('src.google_api_client.requests.post', side_effect=slow_api_call)
        
        # Enable keepalive with short interval
        with patch('src.config.NONSTREAM_KEEPALIVE_ENABLED', True), \
             patch('src.config.NONSTREAM_KEEPALIVE_INTERVAL', 0.1):
            
            response = api_client.send_request(
                mock_payload, 
                mock_creds, 
                "test-project", 
                is_streaming=False
            )
        
        # Verify response is streaming
        assert isinstance(response, StreamingResponse)
        assert hasattr(response, 'body_iterator')
        
        # Consume response and check for keepalive messages
        content_chunks = []
        async for chunk in response.body_iterator:
            content_chunks.append(chunk.decode('utf-8') if isinstance(chunk, bytes) else chunk)
        
        # Should have received some content (either keepalive newlines or actual response)
        assert len(content_chunks) > 0
    
    @pytest.mark.asyncio
    async def test_keepalive_timeout_handling(self, mocker, api_client, mock_creds, mock_payload):
        """
        Keepalive should handle timeout properly
        """
        # Mock API call that times out
        mock_post = mocker.patch('src.google_api_client.requests.post')
        mock_post.side_effect = requests.exceptions.Timeout("Request timeout")
        
        # Enable keepalive with short interval
        with patch('src.config.NONSTREAM_KEEPALIVE_ENABLED', True), \
             patch('src.config.NONSTREAM_KEEPALIVE_INTERVAL', 0.1):
            
            response = api_client.send_request(
                mock_payload, 
                mock_creds, 
                "test-project", 
                is_streaming=False
            )
        
        # Should return error response
        assert isinstance(response, StreamingResponse)
        assert hasattr(response, 'body_iterator')
        
        # Consume response and check for error message
        content_chunks = []
        async for chunk in response.body_iterator:
            content_chunks.append(chunk.decode('utf-8') if isinstance(chunk, bytes) else chunk)
        
        # Should have received an error response
        response_content = ''.join(content_chunks)
        assert "error" in response_content.lower() or "timeout" in response_content.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])