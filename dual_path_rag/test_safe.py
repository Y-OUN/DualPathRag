import os
import sys
import traceback

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置 API Key
os.environ["SILICONFLOW_API_KEY"] = "sk-byuljoelkuljizqeahjtvtobfjvxjgtoofiazzyyllpvluzf"

from core.dual_path_graph import DualPathGraph, DualPathState
from core.performance_monitor import PerformanceMonitor
from langchain_openai import ChatOpenAI


def test_api_key():
    """测试API Key是否设置"""
    api_key = os.environ.get("SILICONFLOW_API_KEY")
    if not api_key:
        print("错误: 未设置 SILICONFLOW_API_KEY 环境变量")
        print("\n请设置环境变量:")
        print("  Windows PowerShell:")
        print("    $env:SILICONFLOW_API_KEY = \"your-api-key\"")
        print("  Windows CMD:")
        print("    set SILICONFLOW_API_KEY=your-api-key")
        print("  Linux/Mac:")
        print("    export SILICONFLOW_API_KEY=your-api-key")
        print("\n获取API Key:")
        print("  1. 访问: `https://cloud.siliconflow.cn` ")
        print("  2. 注册账号并登录")
        print("  3. 点击左侧菜单 'API密钥'")
        print("  4. 点击 '新建 API 密钥'")
        print("  5. 复制生成的密钥")
        return False
    return True


def test_routing_engine():
    """测试路由引擎"""
    print("\n1. 测试路由引擎...")
    try:
        from core.smart_routing import SmartRoutingEngine
        routing_engine = SmartRoutingEngine()
        
        test_queries = [
            "什么是人工智能？",
            "如何实现一个神经网络？",
            "Python的基本语法",
            "深度学习的原理是什么？"
        ]
        
        for query in test_queries:
            result = routing_engine.analyze_query(query)
            print(f"   查询: {query}")
            print(f"   复杂度: {result['complexity']:.2f}")
            print(f"   路由: {result['route']}")
            print(f"   概率: {result['probabilities']}")
            print()
        
        print("   路由引擎测试通过")
        return True
    except Exception as e:
        print(f"   路由引擎测试失败: {e}")
        traceback.print_exc()
        return False


def test_performance_monitor():
    """测试性能监控"""
    print("\n2. 测试性能监控...")
    try:
        monitor = PerformanceMonitor()
        print("   性能监控测试通过")
        return True
    except Exception as e:
        print(f"   性能监控测试失败: {e}")
        traceback.print_exc()
        return False


def test_chat_model():
    """测试聊天模型"""
    print("\n3. 测试聊天模型...")
    try:
        chat_model = ChatOpenAI(
            model="deepseek-ai/DeepSeek-V3",
            openai_api_key=os.environ["SILICONFLOW_API_KEY"],
            openai_api_base="https://api.siliconflow.cn/v1",
            temperature=0.5,
            timeout=30
        )
        
        response = chat_model.invoke([{"role": "user", "content": "你好，请用一句话介绍自己"}])
        print("   聊天成功")
        print(f"   回复: {response.content[:50]}...")
        return True
    except Exception as e:
        print(f"   聊天测试失败: {e}")
        traceback.print_exc()
        return False


def test_dual_path_graph():
    """测试双线架构"""
    print("\n4. 测试双线架构...")
    try:
        graph = DualPathGraph()
        
        test_query = "什么是RAG技术？"
        result = graph.run(test_query)
        
        print(f"   查询: {test_query}")
        print(f"   路由决策: {result['routing_decision']}")
        print(f"   响应: {result['response'][:100]}...")
        print(f"   性能: {result['performance']}")
        
        print("   双线架构测试通过")
        return True
    except Exception as e:
        print(f"   双线架构测试失败: {e}")
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("=" * 60)
    print("智能路由双线架构 - 安全测试")
    print("=" * 60)
    
    # 测试API Key
    if not test_api_key():
        sys.exit(1)
    
    # 运行测试
    tests = [
        test_routing_engine,
        test_performance_monitor,
        test_chat_model,
        test_dual_path_graph
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed}/{total} 通过")
    print("=" * 60)
    
    if passed == total:
        print("所有测试通过！")
        print("\n系统状态:")
        print("  - API Key: 已设置")
        print("  - 路由引擎: 正常")
        print("  - 性能监控: 正常")
        print("  - 聊天模型: 正常")
        return 0
    else:
        print("部分测试失败，请检查错误信息")
        return 1


if __name__ == "__main__":
    sys.exit(main())
