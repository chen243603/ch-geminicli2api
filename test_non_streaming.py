#!/usr/bin/env python3
"""
测试普通非流式请求是否正确运行
"""
import os
import sys
import json
from unittest.mock import MagicMock, patch

# 添加项目根目录到Python路径
sys.path.insert(0, '/Users/pashale/WebstormProjects/geminicli2api')

from src.google_api_client import GoogleApiClient

def test_non_streaming_request():
    """测试非流式请求"""
    print("=== 普通非流式请求测试 ===")
    
    # 创建客户端
    client = GoogleApiClient()
    
    # 模拟API响应
    with patch.object(client, '_make_request') as mock_request:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"response": {"candidates": [{"content": {"parts": [{"text": "测试响应"}]}}]}}'
        mock_response.json.return_value = {"response": {"candidates": [{"content": {"parts": [{"text": "测试响应"}]}}]}}
        mock_request.return_value = mock_response
        
        # 模拟请求参数
        payload = {"contents": [{"parts": [{"text": "测试请求"}]}]}
        creds = MagicMock()
        project_id = "test-project"
        
        print("发送非流式请求...")
        
        # 发送非流式请求
        response = client.send_request(payload, creds, project_id, is_streaming=False)
        
        print(f"响应状态码: {response.status_code}")
        print(f"响应内容: {response.body.decode('utf-8')}")
        
        # 验证请求参数
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        
        # 检查是否使用了正确的端点（非流式）
        call_url = call_args[0][0]
        if "generateContent" in call_url and "streamGenerateContent" not in call_url:
            print("✅ 正确使用了非流式端点")
        else:
            print("❌ 端点选择错误")
        
        # 检查是否使用了stream=False
        stream_param = call_args[1].get('stream', False)
        if not stream_param:
            print("✅ 正确设置了stream=False")
        else:
            print("❌ stream参数设置错误")
        
        # 验证响应格式
        response_data = json.loads(response.body.decode('utf-8'))
        if "candidates" in response_data:
            print("✅ 响应格式正确")
        else:
            print("❌ 响应格式错误")
        
        print("✅ 非流式请求测试完成")

if __name__ == "__main__":
    test_non_streaming_request()