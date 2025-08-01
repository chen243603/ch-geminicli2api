import pytest
import asyncio
import json
import time
from unittest.mock import patch, MagicMock
from src.google_api_client import GoogleApiClient


class TestPseudoStreamingSimple:
    """测试简化的伪流功能"""
    
    @pytest.fixture
    def client(self):
        return GoogleApiClient()

    @pytest.mark.asyncio
    async def test_pseudo_streaming_sends_heartbeats_then_complete_response(self, client):
        """测试伪流模式发送心跳包然后发送完整响应"""
        with patch('src.google_api_client.PSEUDO_STREAMING_ENABLED', True), \
             patch('src.google_api_client.PSEUDO_STREAMING_HEARTBEAT_INTERVAL', 0.1), \
             patch.object(client, '_make_request') as mock_request:
            
            # 模拟完整响应（可能包含额外数据）
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = '''{"response": {"candidates": [{"content": {"parts": [{"text": "Hello world"}]}}]}}
extra data that would cause parsing issues
{"another": "object"}'''
            mock_request.return_value = mock_response
            
            # 模拟请求参数
            payload = {"contents": [{"parts": [{"text": "test"}]}]}
            creds = MagicMock()
            project_id = "test-project"
            
            # 发送流式请求
            streaming_response = client.send_request(payload, creds, project_id, is_streaming=True)
            
            # 收集响应数据
            chunks = []
            start_time = time.time()
            
            async for chunk in streaming_response.body_iterator:
                chunks.append(chunk.decode('utf-8'))
                # 收集所有数据
                if len(chunks) >= 10:  # 防止无限循环
                    break
            
            elapsed_time = time.time() - start_time
            
            # 验证发送了心跳包
            heartbeat_chunks = [chunk for chunk in chunks if chunk.strip() == "data: {}"]
            assert len(heartbeat_chunks) >= 2, f"应该发送2个心跳包，实际收到: {len(heartbeat_chunks)}"
            
            # 验证最后发送了格式化的响应
            response_chunks = [chunk for chunk in chunks if chunk.strip() != "data: {}"]
            assert len(response_chunks) >= 1, "应该包含完整响应"
            
            # 验证响应包含文本内容（经过解析和格式化）
            complete_response = ''.join(response_chunks)
            assert "Hello world" in complete_response, "应该包含响应内容"
            
            # 验证总时间至少等于心跳间隔
            assert elapsed_time >= 0.8, "应该至少等待2个心跳间隔"

    @pytest.mark.asyncio
    async def test_pseudo_streaming_handles_data_prefix(self, client):
        """测试处理带data:前缀的响应"""
        with patch('src.google_api_client.PSEUDO_STREAMING_ENABLED', True), \
             patch('src.google_api_client.PSEUDO_STREAMING_HEARTBEAT_INTERVAL', 0.1), \
             patch.object(client, '_make_request') as mock_request:
            
            # 模拟带data:前缀的响应
            mock_response = MagicMock() 
            mock_response.status_code = 200
            mock_response.text = 'data: {"response": {"candidates": [{"content": {"parts": [{"text": "Test with data prefix"}]}}]}}'
            mock_request.return_value = mock_response
            
            # 模拟请求参数
            payload = {"contents": [{"parts": [{"text": "test"}]}]}
            creds = MagicMock()
            project_id = "test-project"
            
            # 发送流式请求
            streaming_response = client.send_request(payload, creds, project_id, is_streaming=True)
            
            # 收集响应数据
            chunks = []
            async for chunk in streaming_response.body_iterator:
                chunks.append(chunk.decode('utf-8'))
                if len(chunks) >= 10:
                    break
            
            # 验证正确处理了data:前缀
            response_chunks = [chunk for chunk in chunks if chunk.strip() != "data: {}"]
            assert len(response_chunks) >= 1
            
            complete_response = ''.join(response_chunks)
            assert "Test with data prefix" in complete_response

    def test_pseudo_streaming_uses_non_streaming_request(self, client):
        """验证伪流模式使用非流式请求"""
        with patch('src.google_api_client.PSEUDO_STREAMING_ENABLED', True), \
             patch.object(client, '_make_request') as mock_request:
            
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = '{"test": "response"}'
            mock_request.return_value = mock_response
            
            # 发送流式请求
            payload = {"contents": [{"parts": [{"text": "test"}]}]}
            creds = MagicMock()
            project_id = "test-project"
            
            client.send_request(payload, creds, project_id, is_streaming=True)
            
            # 验证使用了非流式请求
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert call_args[1]['stream'] == False, "伪流模式应该发送非流式请求"