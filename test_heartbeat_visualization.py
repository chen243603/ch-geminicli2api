#!/usr/bin/env python3
"""
简单的测试脚本，用于观察伪流模式的心跳包发送情况
"""
import os
import sys
import asyncio
import time
import json
from unittest.mock import MagicMock, patch

# 添加项目根目录到Python路径
sys.path.insert(0, '/Users/pashale/WebstormProjects/geminicli2api')

from src.google_api_client import GoogleApiClient

async def test_heartbeat_visualization():
    """可视化测试心跳包发送"""
    print("=== 伪流心跳包测试 ===")
    
    # 创建客户端
    client = GoogleApiClient()
    
    # 启用伪流模式并设置较短的心跳间隔以便观察
    with patch('src.google_api_client.PSEUDO_STREAMING_ENABLED', True), \
         patch('src.google_api_client.PSEUDO_STREAMING_HEARTBEAT_INTERVAL', 1.0), \
         patch('src.google_api_client.PSEUDO_STREAMING_MAX_HEARTBEATS', 100), \
         patch.object(client, '_make_request') as mock_request:
        
        # 模拟一个较长的响应，并添加延迟
        def delayed_response(*args, **kwargs):
            # 模拟网络延迟
            time.sleep(3)
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.encoding = 'utf-8'
            response_data = {
                "candidates": [{
                    "content": {
                        "role": "model",
                        "parts": [{"text": "这是一个较长的响应内容，用于测试心跳包的发送情况。" * 10}]
                    },
                    "finishReason": "STOP"
                }]
            }
            response_text = json.dumps(response_data, ensure_ascii=False)
            mock_response.text = response_text
            mock_response.content = response_text.encode('utf-8')
            mock_response.json.return_value = response_data
            return mock_response
        
        mock_request.side_effect = delayed_response
        
        # 模拟请求参数
        payload = {"contents": [{"parts": [{"text": "测试心跳包"}]}]}
        creds = MagicMock()
        project_id = "test-project"
        
        print("开始发送流式请求...")
        print("预期行为：")
        print("1. 立即开始发送心跳包")
        print("2. 每1秒发送一个心跳包")
        print("3. 同时处理API响应")
        print("4. 将完整响应分片发送")
        print("5. 最多发送100个心跳包（100秒最大等待时间）")
        print("-" * 50)
        
        # 发送流式请求
        streaming_response = client.send_request(payload, creds, project_id, is_streaming=True)
        
        # 收集响应数据
        chunks = []
        timestamps = []
        heartbeat_count = 0
        response_count = 0
        
        print("开始接收响应...")
        async for chunk in streaming_response.body_iterator:
            chunk_str = chunk.decode('utf-8')
            chunks.append(chunk_str)
            timestamps.append(time.time())
            
            # 分析chunk类型
            if chunk_str.strip() == "data: {}":
                heartbeat_count += 1
                print(f"[{len(chunks):02d}] ❤️  心跳包 #{heartbeat_count}")
            else:
                response_count += 1
                print(f"[{len(chunks):02d}] 📝 响应片段 #{response_count}: {chunk_str[:50]}...")
            
            # 限制收集数量
            if len(chunks) >= 15:
                break
        
        print("-" * 50)
        print("测试结果统计:")
        print(f"总片段数: {len(chunks)}")
        print(f"心跳包数: {heartbeat_count}")
        print(f"响应片段数: {response_count}")
        print(f"总耗时: {timestamps[-1] - timestamps[0]:.2f}秒")
        
        # 验证结果
        if heartbeat_count > 0:
            print("✅ 成功检测到心跳包!")
        else:
            print("❌ 未检测到心跳包!")
        
        # 检查时间间隔
        if len(timestamps) > 1:
            avg_interval = (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1)
            print(f"平均片段间隔: {avg_interval:.2f}秒")
            
            if 0.8 <= avg_interval <= 1.2:  # 允许一些误差
                print("✅ 心跳间隔符合预期!")
            else:
                print("❌ 心跳间隔不符合预期!")
        
        # 显示完整的响应内容
        print("\n完整响应内容:")
        for i, chunk in enumerate(chunks):
            print(f"片段 {i+1}: {chunk}")

if __name__ == "__main__":
    asyncio.run(test_heartbeat_visualization())