import pytest
import os
from unittest.mock import patch

from src.config import (
    get_base_model_name,
    is_search_model,
    is_nothinking_model,
    is_maxthinking_model,
    get_thinking_budget,
    should_include_thoughts,
    NONSTREAM_KEEPALIVE_ENABLED,
    NONSTREAM_KEEPALIVE_INTERVAL
)


class TestModelHelpers:
    """Test model name helper functions."""
    
    def test_get_base_model_name_with_variants(self):
        """Test that base model names are correctly extracted from variants."""
        assert get_base_model_name("models/gemini-2.5-pro") == "models/gemini-2.5-pro"
        assert get_base_model_name("models/gemini-2.5-pro-search") == "models/gemini-2.5-pro"
        assert get_base_model_name("models/gemini-2.5-pro-nothinking") == "models/gemini-2.5-pro"
        assert get_base_model_name("models/gemini-2.5-pro-maxthinking") == "models/gemini-2.5-pro"
        # Note: The current implementation removes the first matching suffix only
        assert get_base_model_name("models/gemini-2.5-pro-search-nothinking") == "models/gemini-2.5-pro-search"
        assert get_base_model_name("models/gemini-2.5-pro-search-maxthinking") == "models/gemini-2.5-pro-search"
    
    def test_is_search_model(self):
        """Test search model detection."""
        assert is_search_model("models/gemini-2.5-pro-search") is True
        assert is_search_model("models/gemini-2.5-pro-search-nothinking") is True
        assert is_search_model("models/gemini-2.5-pro") is False
        assert is_search_model("models/gemini-2.5-pro-nothinking") is False
    
    def test_is_nothinking_model(self):
        """Test nothinking model detection."""
        assert is_nothinking_model("models/gemini-2.5-pro-nothinking") is True
        assert is_nothinking_model("models/gemini-2.5-pro-search-nothinking") is True
        assert is_nothinking_model("models/gemini-2.5-pro") is False
        assert is_nothinking_model("models/gemini-2.5-pro-search") is False
    
    def test_is_maxthinking_model(self):
        """Test maxthinking model detection."""
        assert is_maxthinking_model("models/gemini-2.5-pro-maxthinking") is True
        assert is_maxthinking_model("models/gemini-2.5-pro-search-maxthinking") is True
        assert is_maxthinking_model("models/gemini-2.5-pro") is False
        assert is_maxthinking_model("models/gemini-2.5-pro-nothinking") is False
    
    def test_get_thinking_budget(self):
        """Test thinking budget calculation."""
        # Default models
        assert get_thinking_budget("models/gemini-2.5-pro") == -1
        assert get_thinking_budget("models/gemini-2.5-flash") == -1
        
        # Nothinking variants
        assert get_thinking_budget("models/gemini-2.5-flash-nothinking") == 0
        assert get_thinking_budget("models/gemini-2.5-pro-nothinking") == 128
        
        # Maxthinking variants
        assert get_thinking_budget("models/gemini-2.5-flash-maxthinking") == 24576
        assert get_thinking_budget("models/gemini-2.5-pro-maxthinking") == 32768
    
    def test_should_include_thoughts(self):
        """Test thoughts inclusion logic."""
        # Default models should include thoughts
        assert should_include_thoughts("models/gemini-2.5-pro") is True
        assert should_include_thoughts("models/gemini-2.5-flash") is True
        
        # Maxthinking models should include thoughts
        assert should_include_thoughts("models/gemini-2.5-pro-maxthinking") is True
        assert should_include_thoughts("models/gemini-2.5-flash-maxthinking") is True
        
        # Nothinking variants - only pro should include thoughts
        assert should_include_thoughts("models/gemini-2.5-pro-nothinking") is True
        assert should_include_thoughts("models/gemini-2.5-flash-nothinking") is False


class TestKeepaliveConfig:
    """Test keepalive configuration."""
    
    def test_keepalive_enabled_from_env(self):
        """Test that keepalive can be enabled via environment variable."""
        with patch.dict(os.environ, {'NONSTREAM_KEEPALIVE_ENABLED': 'true'}):
            # Reload the config module to pick up new env vars
            import importlib
            import src.config
            importlib.reload(src.config)
            assert src.config.NONSTREAM_KEEPALIVE_ENABLED is True
    
    def test_keepalive_disabled_by_default(self):
        """Test that keepalive is disabled by default."""
        with patch.dict(os.environ, {}, clear=True):
            # Reload the config module to pick up new env vars
            import importlib
            import src.config
            importlib.reload(src.config)
            assert src.config.NONSTREAM_KEEPALIVE_ENABLED is False
    
    def test_keepalive_interval_from_env(self):
        """Test that keepalive interval can be set via environment variable."""
        with patch.dict(os.environ, {'NONSTREAM_KEEPALIVE_INTERVAL': '10.5'}):
            # Reload the config module to pick up new env vars
            import importlib
            import src.config
            importlib.reload(src.config)
            assert src.config.NONSTREAM_KEEPALIVE_INTERVAL == 10.5
    
    def test_keepalive_interval_default(self):
        """Test default keepalive interval."""
        with patch.dict(os.environ, {}, clear=True):
            # Reload the config module to pick up new env vars
            import importlib
            import src.config
            importlib.reload(src.config)
            assert src.config.NONSTREAM_KEEPALIVE_INTERVAL == 5.0