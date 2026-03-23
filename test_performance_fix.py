import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dual_path_rag'))

from core.dual_path_graph import DualPathGraph


def test_performance_fix():
    """测试性能指标修复"""
    print("测试性能指标修复...")
    print("=" * 80)
    
    graph = DualPathGraph()
    
    test_queries = [
        "什么是人工智能？",
        "Python的基本语法",
        "如何实现一个神经网络？",
        "深度学习的原理是什么？"
    ]
    
    for query in test_queries:
        print(f"\n查询: {query}")
        
        result = graph.run(query)
        
        print(f"路由决策: {result['routing_decision']}")
        print(f"响应: {result['response'][:100]}...")
        print(f"性能指标:")
        
        performance = result['performance']
        for key, value in performance.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        # 验证时间指标
        total_time = performance['total_time']
        routing_time = performance['routing_time']
        rag_time = performance['rag_time']
        fine_tuned_time = performance['fine_tuned_time']
        merge_time = performance['merge_time']
        
        print(f"\n时间验证:")
        print(f"  路由时间: {routing_time:.4f}秒")
        if result['routing_decision'] == 'rag':
            print(f"  RAG时间: {rag_time:.4f}秒")
            print(f"  微调时间: {fine_tuned_time:.4f}秒 (应为0)")
        else:
            print(f"  RAG时间: {rag_time:.4f}秒 (应为0)")
            print(f"  微调时间: {fine_tuned_time:.4f}秒")
        print(f"  合并时间: {merge_time:.4f}秒")
        print(f"  总时间: {total_time:.4f}秒")
    
    graph.shutdown()
    print("\n" + "=" * 80)
    print("测试完成！")


if __name__ == "__main__":
    test_performance_fix()
