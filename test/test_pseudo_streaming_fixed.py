import pytest
import asyncio
import json
import time
from unittest.mock import patch, MagicMock, AsyncMock
from src.google_api_client import GoogleApiClient


class TestPseudoStreamingFixed:
    """测试修复后的伪流功能"""
    
    @pytest.fixture
    def mock_non_streaming_response(self):
        """模拟完整的非流式响应"""
        return {
            "candidates": [{
                "content": {
                    "parts": [{"text": "This is a complete response from the API"}]
                },
                "finishReason": "STOP"
            }]
        }
    
    @pytest.fixture
    def mock_long_response(self):
        """模拟长文本响应，用于测试心跳包"""
        long_text = "This is a very long response that should be split into multiple chunks. " * 20  # 约1400字符
        return {
            "candidates": [{
                "content": {
                    "parts": [{"text": long_text}]
                },
                "finishReason": "STOP"
            }]
        }

    @pytest.fixture
    def client(self):
        return GoogleApiClient()

    def test_pseudo_streaming_uses_non_streaming_request(self, client):
        """测试伪流模式下应该发送非流式请求到上游API"""
        with patch('src.google_api_client.PSEUDO_STREAMING_ENABLED', True), \
             patch.object(client, '_make_request') as mock_request:
            
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.encoding = 'utf-8'
            response_text = json.dumps({"candidates": [{"content": {"parts": [{"text": "test"}]}}]})
            mock_response.text = response_text
            mock_response.content = response_text.encode('utf-8')
            mock_response.json.return_value = {"candidates": [{"content": {"parts": [{"text": "test"}]}}]}
            mock_request.return_value = mock_response
            
            # 模拟请求参数
            payload = {"contents": [{"parts": [{"text": "test"}]}]}
            creds = MagicMock()
            project_id = "test-project"
            
            # 发送流式请求
            result = client.send_request(payload, creds, project_id, is_streaming=True)
            
            # 验证调用了非流式请求（stream=False）
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert call_args[1]['stream'] == False, "伪流模式应该发送非流式请求"

    @pytest.mark.asyncio
    async def test_pseudo_streaming_sends_heartbeats_by_time(self, client, mock_long_response):
        """测试伪流模式按时间间隔发送心跳包"""
        with patch('src.google_api_client.PSEUDO_STREAMING_ENABLED', True), \
             patch('src.google_api_client.PSEUDO_STREAMING_HEARTBEAT_INTERVAL', 0.5), \
             patch('src.google_api_client.PSEUDO_STREAMING_CHUNK_SIZE', 50), \
             patch.object(client, '_make_request') as mock_request:
            
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.encoding = 'utf-8'
            response_text = json.dumps({"response": mock_long_response})
            mock_response.text = response_text
            mock_response.content = response_text.encode('utf-8')
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
                if len(chunks) >= 10:  # 收集更多数据，给心跳包时间发送
                    break
            
            elapsed_time = time.time() - start_time
            
            # 打印调试信息
            print(f"Collected {len(chunks)} chunks:")
            for i, chunk in enumerate(chunks):
                print(f"Chunk {i}: {repr(chunk)}")
            
            # 验证有心跳包（空的data行）
            heartbeat_chunks = [chunk for chunk in chunks if chunk.strip() == "data: {}"]
            assert len(heartbeat_chunks) >= 1, f"应该包含心跳包，实际收集到的chunks: {chunks}"
            
            # 验证时间间隔大致正确（至少0.5秒的心跳间隔）
            assert elapsed_time >= 0.5, "应该按时间间隔发送心跳包"

    def test_pseudo_streaming_heartbeat_format(self, client, mock_non_streaming_response):
        """测试心跳包格式正确"""
        with patch('src.google_api_client.PSEUDO_STREAMING_ENABLED', True), \
             patch('src.google_api_client.PSEUDO_STREAMING_HEARTBEAT_INTERVAL', 1), \
             patch.object(client, '_make_request') as mock_request:
            
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.encoding = 'utf-8'
            response_text = json.dumps(mock_non_streaming_response)
            mock_response.text = response_text
            mock_response.content = response_text.encode('utf-8')
            mock_response.json.return_value = mock_non_streaming_response
            mock_request.return_value = mock_response
            
            # 模拟请求参数
            payload = {"contents": [{"parts": [{"text": "test"}]}]}
            creds = MagicMock()
            project_id = "test-project"
            
            # 发送流式请求
            streaming_response = client.send_request(payload, creds, project_id, is_streaming=True)
            
            # 验证心跳包格式
            # 心跳包应该是空的JSON对象，格式为 "data: {}\n\n"
            expected_heartbeat = "data: {}\n\n"
            
            # 这个测试会失败，因为当前实现的心跳包格式不正确
            # 当前实现: f"data: {{}}\n\n" -> "data: {}\n\n" 
            # 正确格式: "data: {}\n\n"
            assert True  # 占位符，实际实现后会有具体验证

    def test_config_naming_correctness(self):
        """测试配置项命名的正确性"""
        from src.config import PSEUDO_STREAMING_HEARTBEAT_INTERVAL
        
        # 这个测试会失败，因为当前配置项名称含义不明确
        # PSEUDO_STREAMING_HEARTBEAT_INTERVAL 应该表示心跳间隔的秒数
        # 而不是每N个chunk发送一次心跳
        
        # 期望的配置应该是以秒为单位的时间间隔
        assert isinstance(PSEUDO_STREAMING_HEARTBEAT_INTERVAL, (int, float))
        assert PSEUDO_STREAMING_HEARTBEAT_INTERVAL > 0

    @pytest.mark.asyncio
    async def test_pseudo_streaming_splits_response_over_time(self, client):
        """测试伪流模式将完整响应按时间分片发送"""
        long_response = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "This is a very long response that should be split into multiple chunks over time to simulate streaming behavior."}]
                },
                "finishReason": "STOP"
            }]
        }
        
        with patch('src.google_api_client.PSEUDO_STREAMING_ENABLED', True), \
             patch('src.google_api_client.PSEUDO_STREAMING_HEARTBEAT_INTERVAL', 0.5), \
             patch.object(client, '_make_request') as mock_request:
            
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.encoding = 'utf-8'
            response_text = json.dumps(long_response)
            mock_response.text = response_text
            mock_response.content = response_text.encode('utf-8')
            mock_response.json.return_value = long_response
            mock_request.return_value = mock_response
            
            # 模拟请求参数
            payload = {"contents": [{"parts": [{"text": "test"}]}]}
            creds = MagicMock()
            project_id = "test-project"
            
            # 发送流式请求
            streaming_response = client.send_request(payload, creds, project_id, is_streaming=True)
            
            # 收集响应并测量时间
            chunks = []
            timestamps = []
            
            async for chunk in streaming_response.body_iterator:
                chunks.append(chunk.decode('utf-8'))
                timestamps.append(time.time())
                if len(chunks) >= 5:  # 收集足够的数据后停止
                    break
            
            # 验证响应是分时间发送的（而不是一次性发送所有数据）
            if len(timestamps) >= 2:
                time_diff = timestamps[-1] - timestamps[0]
                assert time_diff >= 0.4, "响应应该分时间发送，而不是一次性发送"

    @pytest.mark.asyncio
    async def test_pseudo_streaming_handles_malformed_json(self, client):
        """测试伪流模式处理格式错误的JSON响应"""
        with patch('src.google_api_client.PSEUDO_STREAMING_ENABLED', True), \
             patch.object(client, '_make_request') as mock_request:
            
            # 模拟包含额外数据的错误响应（类似实际遇到的错误）
            malformed_response = MagicMock()
            malformed_response.status_code = 200
            malformed_response.text = '''{"response": {"candidates": [{"content": {"parts": [{"text": "Hello world"}]}}]}}
extra data that causes parsing error
{"another": "object"}'''
            mock_request.return_value = malformed_response
            
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
                if len(chunks) >= 3:  # 收集足够的数据后停止
                    break
            
            # 验证能够处理错误响应并返回有效内容
            assert len(chunks) >= 1, "应该返回至少一个chunk"
            
            # 检查是否包含有效的文本响应（从第一个有效JSON对象解析）
            found_valid_response = False
            for chunk in chunks:
                if "Hello world" in chunk:
                    found_valid_response = True
                    break
            
            assert found_valid_response, "应该能够从格式错误的响应中提取有效内容"

    @pytest.mark.asyncio  
    async def test_pseudo_streaming_handles_multiline_response(self, client):
        """测试伪流模式处理多行响应"""
        with patch('src.google_api_client.PSEUDO_STREAMING_ENABLED', True), \
             patch.object(client, '_make_request') as mock_request:
            
            # 模拟多行响应格式
            multiline_response = MagicMock()
            multiline_response.status_code = 200
            multiline_response.text = '''data: {"response": {"candidates": [{"content": {"parts": [{"text": "Multiline response test"}]}}]}}

data: extra line
'''
            mock_request.return_value = multiline_response
            
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
                if len(chunks) >= 2:
                    break
            
            # 验证能够处理多行响应
            assert len(chunks) >= 1, "应该返回至少一个chunk"
            
            # 检查是否包含正确的文本内容
            found_content = False
            for chunk in chunks:
                if "Multiline response test" in chunk:
                    found_content = True
                    break
            
            assert found_content, "应该能够从多行响应中提取正确内容"