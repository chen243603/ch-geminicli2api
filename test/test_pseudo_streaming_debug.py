import pytest
import asyncio
import json
import logging
from unittest.mock import patch, MagicMock
from src.google_api_client import GoogleApiClient


class TestPseudoStreamingDebug:
    """测试伪流功能的调试和日志输出"""
    
    @pytest.fixture
    def client(self):
        return GoogleApiClient()

    @pytest.mark.asyncio
    async def test_pseudo_streaming_logs_raw_response(self, client, caplog):
        """测试伪流模式记录原始响应内容"""
        with patch('src.google_api_client.PSEUDO_STREAMING_ENABLED', True), \
             patch('src.google_api_client.PSEUDO_STREAMING_HEARTBEAT_INTERVAL', 0.1), \
             patch.object(client, '_make_request') as mock_request:
            
            # 设置日志级别
            caplog.set_level(logging.INFO)
            
            # 模拟包含额外数据的响应
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.encoding = 'utf-8'
            response_text = '''{"response": {"candidates": [{"content": {"parts": [{"text": "Debug test response"}]}}]}}
extra data line
{"another": "json", "object": "here"}'''
            mock_response.text = response_text
            mock_response.content = response_text.encode('utf-8')
            mock_request.return_value = mock_response
            
            # 模拟请求参数
            payload = {"contents": [{"parts": [{"text": "debug test"}]}]}
            creds = MagicMock()
            project_id = "test-project"
            
            # 发送流式请求
            streaming_response = client.send_request(payload, creds, project_id, is_streaming=True)
            
            # 收集响应数据
            chunks = []
            async for chunk in streaming_response.body_iterator:
                chunks.append(chunk.decode('utf-8'))
                if len(chunks) >= 5:
                    break
            
            # 验证日志包含原始响应内容
            log_messages = [record.message for record in caplog.records]
            raw_response_logged = any("Pseudo streaming raw response:" in msg for msg in log_messages)
            assert raw_response_logged, f"应该记录原始响应内容，实际日志: {log_messages}"
            
            # 验证响应包含正确内容
            complete_response = ''.join(chunks)
            assert "Debug test response" in complete_response, "应该包含测试响应内容"

    @pytest.mark.asyncio
    async def test_pseudo_streaming_handles_parsing_error(self, client, caplog):
        """测试伪流模式处理JSON解析错误"""
        with patch('src.google_api_client.PSEUDO_STREAMING_ENABLED', True), \
             patch('src.google_api_client.PSEUDO_STREAMING_HEARTBEAT_INTERVAL', 0.1), \
             patch.object(client, '_make_request') as mock_request:
            
            # 设置日志级别
            caplog.set_level(logging.ERROR)
            
            # 模拟完全无法解析的响应
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.encoding = 'utf-8'
            response_text = '''invalid json content here
no valid json at all
just some random text'''
            mock_response.text = response_text
            mock_response.content = response_text.encode('utf-8')
            mock_request.return_value = mock_response
            
            # 模拟请求参数
            payload = {"contents": [{"parts": [{"text": "error test"}]}]}
            creds = MagicMock()
            project_id = "test-project"
            
            # 发送流式请求
            streaming_response = client.send_request(payload, creds, project_id, is_streaming=True)
            
            # 收集响应数据
            chunks = []
            async for chunk in streaming_response.body_iterator:
                chunks.append(chunk.decode('utf-8'))
                if len(chunks) >= 5:
                    break
            
            # 验证错误日志
            log_messages = [record.message for record in caplog.records]
            parse_error_logged = any("Failed to parse response in pseudo streaming:" in msg for msg in log_messages)
            raw_content_logged = any("Raw response content:" in msg for msg in log_messages)
            
            assert parse_error_logged, f"应该记录解析错误，实际日志: {log_messages}"
            assert raw_content_logged, f"应该记录原始响应内容，实际日志: {log_messages}"
            
            # 验证返回了错误响应
            complete_response = ''.join(chunks)
            assert "parse_error" in complete_response or "Failed to parse" in complete_response, "应该返回解析错误信息"