import pytest
import os
import json
from unittest.mock import patch, MagicMock
from src.google_api_client import GoogleApiClient

def test_pseudo_streaming_response_handling():
    """测试伪流功能的响应处理"""
    # Enable pseudo streaming for this test
    os.environ["PSEUDO_STREAMING_ENABLED"] = "true"
    os.environ["PSEUDO_STREAMING_HEARTBEAT_INTERVAL"] = "2"
    
    # Reload config
    import importlib
    import src.config
    importlib.reload(src.config)
    from src.config import PSEUDO_STREAMING_ENABLED, PSEUDO_STREAMING_HEARTBEAT_INTERVAL
    
    # Verify config is loaded correctly
    assert PSEUDO_STREAMING_ENABLED == True
    assert PSEUDO_STREAMING_HEARTBEAT_INTERVAL == 2
    
    # Create a GoogleApiClient instance
    client = GoogleApiClient()
    
    # Create a mock response with multiple chunks
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_lines.return_value = [
        b'data: {"response": {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}}',
        b'data: {"response": {"candidates": [{"content": {"parts": [{"text": " world"}]}}]}}',
        b'data: {"response": {"candidates": [{"content": {"parts": [{"text": "!"}]}}]}}',
        b'data: {"response": {"candidates": [{"content": {"parts": [{"text": " How"}]}}]}}',
        b'data: {"response": {"candidates": [{"content": {"parts": [{"text": " are"}]}}]}}',
        b'data: {"response": {"candidates": [{"content": {"parts": [{"text": " you?"}]}}]}}',
    ]
    
    # Test the streaming response handling
    streaming_response = client._handle_streaming_response(mock_response)
    
    # Verify it's a StreamingResponse
    assert streaming_response is not None
    assert hasattr(streaming_response, 'body_iterator')
    
    # Restore environment
    os.environ.pop("PSEUDO_STREAMING_ENABLED", None)
    os.environ.pop("PSEUDO_STREAMING_HEARTBEAT_INTERVAL", None)
    importlib.reload(src.config)