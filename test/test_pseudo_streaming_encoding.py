import pytest
import asyncio
import json
from unittest.mock import patch, MagicMock
from src.google_api_client import GoogleApiClient


class TestPseudoStreamingEncoding:
    """测试伪流模式的编码处理"""
    
    @pytest.fixture
    def client(self):
        return GoogleApiClient()

    @pytest.mark.asyncio
    async def test_pseudo_streaming_chinese_text(self, client):
        """测试伪流模式处理中文文本"""
        with patch('src.google_api_client.PSEUDO_STREAMING_ENABLED', True), \
             patch('src.google_api_client.PSEUDO_STREAMING_HEARTBEAT_INTERVAL', 0.1), \
             patch.object(client, '_make_request') as mock_request:
            
            # 模拟包含中文的流式响应
            chinese_text = "这是一段中文测试文本，用来验证伪流模式的编码处理是否正确。"
            response_data = {
                "response": {
                    "candidates": [{
                        "content": {
                            "role": "model",
                            "parts": [{"text": chinese_text}]
                        }
                    }]
                }
            }
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = f"data: {json.dumps(response_data, ensure_ascii=False)}"
            mock_request.return_value = mock_response
            
            # 模拟请求参数
            payload = {"contents": [{"parts": [{"text": "请用中文回答"}]}]}
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
            
            # 验证中文内容正确编码
            complete_response = ''.join(chunks)
            assert chinese_text in complete_response, f"中文内容应该正确编码，实际响应: {complete_response}"
            
            # 验证JSON格式正确
            response_chunks = [chunk for chunk in chunks if chunk.strip() != "data: {}"]
            assert len(response_chunks) >= 1, "应该包含至少一个响应chunk"
            
            # 尝试解析JSON验证格式正确性
            for chunk in response_chunks:
                if chunk.startswith("data: ") and chunk.strip() != "data: {}":
                    json_part = chunk[6:].strip()
                    try:
                        parsed = json.loads(json_part)
                        assert "candidates" in parsed, "响应应该包含candidates字段"
                        break
                    except json.JSONDecodeError as e:
                        pytest.fail(f"JSON解析失败: {e}, chunk: {chunk}")

    @pytest.mark.asyncio
    async def test_pseudo_streaming_mixed_content(self, client):
        """测试伪流模式处理混合内容（思考+文本）"""
        with patch('src.google_api_client.PSEUDO_STREAMING_ENABLED', True), \
             patch('src.google_api_client.PSEUDO_STREAMING_HEARTBEAT_INTERVAL', 0.1), \
             patch.object(client, '_make_request') as mock_request:
            
            # 模拟包含思考和文本的流式响应
            thought_text = "让我思考一下这个问题..."
            response_text = "这是我的回答：中文处理测试成功。"
            
            thought_data = {
                "response": {
                    "candidates": [{
                        "content": {
                            "role": "model",
                            "parts": [{"thought": True, "text": thought_text}]
                        }
                    }]
                }
            }
            
            response_data = {
                "response": {
                    "candidates": [{
                        "content": {
                            "role": "model",
                            "parts": [{"text": response_text}]
                        }
                    }]
                }
            }
            
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = f"data: {json.dumps(thought_data, ensure_ascii=False)}\n\ndata: {json.dumps(response_data, ensure_ascii=False)}"
            mock_request.return_value = mock_response
            
            # 模拟请求参数
            payload = {"contents": [{"parts": [{"text": "测试请求"}]}]}
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
            
            # 验证混合内容正确处理
            complete_response = ''.join(chunks)
            assert thought_text in complete_response, f"思考内容应该被包含，实际响应: {complete_response}"
            assert response_text in complete_response, f"回答内容应该被包含，实际响应: {complete_response}"