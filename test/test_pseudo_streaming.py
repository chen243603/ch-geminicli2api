import pytest
import os
import json
from unittest.mock import patch, MagicMock

def test_pseudo_streaming_disabled_by_default():
    """测试默认情况下伪流功能是禁用的"""
    # Import here to get fresh values
    from src.config import PSEUDO_STREAMING_ENABLED
    # 检查环境变量默认值
    assert PSEUDO_STREAMING_ENABLED == False

def test_pseudo_streaming_enabled_when_env_var_set():
    """测试当环境变量设置时启用伪流功能"""
    # Backup original value
    original_value = os.environ.get("PSEUDO_STREAMING_ENABLED")
    
    # Set environment variable to enable pseudo streaming
    os.environ["PSEUDO_STREAMING_ENABLED"] = "true"
    
    # Reload the config module to get updated value
    import importlib
    import src.config
    importlib.reload(src.config)
    from src.config import PSEUDO_STREAMING_ENABLED
    
    # Check that pseudo streaming is enabled
    assert PSEUDO_STREAMING_ENABLED == True
    
    # Restore original value
    if original_value is not None:
        os.environ["PSEUDO_STREAMING_ENABLED"] = original_value
    else:
        os.environ.pop("PSEUDO_STREAMING_ENABLED", None)
        
    # Reload again to restore original state
    importlib.reload(src.config)

@patch('src.google_api_client.get_google_api_client')
def test_pseudo_streaming_functionality(mock_get_client):
    """测试伪流功能的具体实现"""
    # Import here to get the updated config
    from src.config import PSEUDO_STREAMING_ENABLED
    
    # Test with pseudo streaming disabled (default behavior)
    assert PSEUDO_STREAMING_ENABLED == False
    
    # Test with pseudo streaming enabled
    os.environ["PSEUDO_STREAMING_ENABLED"] = "true"
    os.environ["PSEUDO_STREAMING_HEARTBEAT_INTERVAL"] = "2"  # Send heartbeat every 2 chunks
    
    # Reload config
    import importlib
    import src.config
    importlib.reload(src.config)
    from src.config import PSEUDO_STREAMING_ENABLED, PSEUDO_STREAMING_HEARTBEAT_INTERVAL
    assert PSEUDO_STREAMING_ENABLED == True
    assert PSEUDO_STREAMING_HEARTBEAT_INTERVAL == 2
    
    # Restore environment
    os.environ.pop("PSEUDO_STREAMING_ENABLED", None)
    os.environ.pop("PSEUDO_STREAMING_HEARTBEAT_INTERVAL", None)
    importlib.reload(src.config)

def test_heartbeat_interval_default():
    """测试心跳间隔的默认值"""
    # Import to get fresh values
    from src.config import PSEUDO_STREAMING_HEARTBEAT_INTERVAL
    
    # Check default value
    assert PSEUDO_STREAMING_HEARTBEAT_INTERVAL == 5.0
    
    # Test custom value
    original_value = os.environ.get("PSEUDO_STREAMING_HEARTBEAT_INTERVAL")
    os.environ["PSEUDO_STREAMING_HEARTBEAT_INTERVAL"] = "10"
    
    # Reload config
    import importlib
    import src.config
    importlib.reload(src.config)
    from src.config import PSEUDO_STREAMING_HEARTBEAT_INTERVAL
    assert PSEUDO_STREAMING_HEARTBEAT_INTERVAL == 10
    
    # Restore original value
    if original_value is not None:
        os.environ["PSEUDO_STREAMING_HEARTBEAT_INTERVAL"] = original_value
    else:
        os.environ.pop("PSEUDO_STREAMING_HEARTBEAT_INTERVAL", None)
        
    # Reload again to restore original state
    importlib.reload(src.config)