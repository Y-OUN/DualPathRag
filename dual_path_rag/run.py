import os
import sys
import time

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.dual_path_graph import DualPathGraph


def main():
    """主函数"""
    print("智能路由双线架构 - 运行")
    print("=" * 60)
    
    # 检查API Key
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
        return 1
    
    # 初始化双线架构
    print("初始化智能路由双线架构...")
    graph = DualPathGraph()
    
    print("\n系统已就绪，输入 'exit' 退出")
    print("=" * 60)
    
    try:
        while True:
            # 获取用户输入
            user_input = input("\n用户: ")
            
            if user_input.lower() == 'exit':
                break
            
            if not user_input.strip():
                continue
            
            # 运行智能路由
            print("\nAI: 思考中...")
            start_time = time.time()
            
            result = graph.run(user_input)
            
            end_time = time.time()
            
            # 显示结果
            print(f"\nAI: {result['response']}")
            print(f"\n路由决策: {result['routing_decision']}")
            print(f"响应时间: {end_time - start_time:.2f} 秒")
            print(f"性能指标: {result['performance']}")
            
            if result['retrieved_documents']:
                print("\n检索到的文档:")
                for i, doc in enumerate(result['retrieved_documents'][:3], 1):
                    print(f"  {i}. {doc['content'][:100]}...")
    
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    finally:
        # 关闭资源
        graph.shutdown()
        print("\n系统已关闭")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
