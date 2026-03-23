from typing import Optional
import re
import nltk
from nltk.corpus import stopwords
import os

# 确保nltk数据可用
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class SmartRoutingEngine:
    """智能路由引擎"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('chinese'))
        self.complexity_keywords = [
            "如何", "为什么", "原理", "机制", "算法", "公式", "证明",
            "分析", "研究", "探讨", "深入", "详细", "全面", "系统"
        ]
        self.simple_keywords = [
            "是什么", "有什么", "在哪里", "什么时候", "谁", "哪个",
            "多少", "如何使用", "怎么操作", "步骤", "方法", "教程"
        ]
    
    def decide_route(self, query: str) -> str:
        """决定路由路径"""
        complexity_score = self._calculate_complexity(query)
        
        if complexity_score > 0.4:
            return "fine_tuned"
        else:
            return "rag"
    
    def _calculate_complexity(self, query: str) -> float:
        """计算查询复杂度"""
        if not query:
            return 0.0
        
        # 基础复杂度
        complexity = 0.0
        
        # 1. 长度分析
        query_length = len(query)
        if query_length > 100:
            complexity += 0.3
        elif query_length > 50:
            complexity += 0.15
        
        # 2. 关键词分析
        complex_keyword_count = 0
        simple_keyword_count = 0
        
        for keyword in self.complexity_keywords:
            if keyword in query:
                complex_keyword_count += 1
        
        for keyword in self.simple_keywords:
            if keyword in query:
                simple_keyword_count += 1
        
        # 复杂关键词加分
        complexity += min(complex_keyword_count * 0.2, 0.4)
        
        # 简单关键词减分
        complexity -= min(simple_keyword_count * 0.1, 0.2)
        
        # 3. 语法结构分析
        if re.search(r'[？?]+$', query):
            complexity += 0.1
        
        if re.search(r'(因为|所以|如果|那么|虽然|但是|不仅|而且)', query):
            complexity += 0.15
        
        # 4. 专业术语分析
        technical_terms = [
            "机器学习", "深度学习", "神经网络", "算法", "模型", "训练",
            "验证", "测试", "精度", "召回率", "F1值", "准确率",
            "向量", "嵌入", "检索", "生成", "微调", "预训练"
        ]
        
        technical_count = 0
        for term in technical_terms:
            if term in query:
                technical_count += 1
        
        complexity += min(technical_count * 0.1, 0.25)
        
        # 确保复杂度在0-1之间
        return max(0.0, min(1.0, complexity))
    
    def get_route_probability(self, query: str) -> dict:
        """获取路由概率"""
        complexity = self._calculate_complexity(query)
        
        return {
            "rag": max(0.0, min(1.0, 1.0 - complexity)),
            "fine_tuned": max(0.0, min(1.0, complexity))
        }
    
    def analyze_query(self, query: str) -> dict:
        """分析查询"""
        return {
            "query": query,
            "complexity": self._calculate_complexity(query),
            "route": self.decide_route(query),
            "probabilities": self.get_route_probability(query)
        }
