from typing import Dict, Any, Optional, List
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
import time
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from .parallel_rag import ParallelRAG
from .performance_monitor import PerformanceMonitor, PerformanceMetrics
from .smart_routing import SmartRoutingEngine


class DualPathState(BaseModel):
    """双线架构状态"""
    user_query: str = Field(..., description="用户查询")
    routing_decision: str = Field(default="rag", description="路由决策")
    rag_response: Optional[str] = Field(default=None, description="RAG路径响应")
    fine_tuned_response: Optional[str] = Field(default=None, description="微调路径响应")
    final_response: Optional[str] = Field(default=None, description="最终响应")
    performance_metrics: PerformanceMetrics = Field(default_factory=PerformanceMetrics, description="性能指标")
    retrieved_documents: List[Dict[str, Any]] = Field(default_factory=list, description="检索到的文档")


class DualPathGraph:
    """智能路由双线架构"""
    
    def __init__(self, state_type=DualPathState):
        self.state_type = state_type
        self.graph = self._build_graph()
        self.compiled_graph = self.graph.compile()
        self.rag = ParallelRAG()
        self.performance_monitor = PerformanceMonitor()
        self.smart_routing = SmartRoutingEngine()
        self._embedding_initialized = False
        self.embedding_model = None
    
    def _build_graph(self) -> StateGraph:
        """构建状态图"""
        graph = StateGraph(self.state_type)
        
        # 添加节点
        graph.add_node("route", self._route_query)
        graph.add_node("process_rag", self._process_rag)
        graph.add_node("process_fine_tuned", self._process_fine_tuned)
        graph.add_node("merge_responses", self._merge_responses)
        
        # 添加边
        graph.add_edge(START, "route")
        graph.add_conditional_edges(
            "route",
            self._decide_route,
            {
                "rag": "process_rag",
                "fine_tuned": "process_fine_tuned"
            }
        )
        graph.add_edge("process_rag", "merge_responses")
        graph.add_edge("process_fine_tuned", "merge_responses")
        graph.add_edge("merge_responses", END)
        
        return graph
    
    def _init_embedding_model(self):
        """延迟初始化嵌入模型"""
        if self._embedding_initialized:
            return
        
        self._embedding_initialized = True
        
        try:
            self.embedding_model = OpenAIEmbeddings(
                model="BAAI/bge-m3",
                openai_api_key=os.getenv("SILICONFLOW_API_KEY"),
                openai_api_base="https://api.siliconflow.cn/v1"
            )
        except Exception as e:
            print(f"警告: 无法加载硅基流动嵌入模型: {e}")
            self.embedding_model = None
    
    def _route_query(self, state: DualPathState) -> Dict[str, Any]:
        """路由查询"""
        start_time = time.time()
        
        # 初始化嵌入模型
        self._init_embedding_model()
        
        # 智能路由决策
        decision = self.smart_routing.decide_route(state.user_query)
        
        routing_time = time.time() - start_time
        
        return {
            "routing_decision": decision,
            "performance_metrics": PerformanceMetrics(
                routing_time=routing_time,
                total_time=routing_time
            )
        }
    
    def _decide_route(self, state: DualPathState) -> str:
        """决定路由路径"""
        return state.routing_decision
    
    def _process_rag(self, state: DualPathState) -> Dict[str, Any]:
        """处理RAG路径"""
        start_time = time.time()
        
        # 初始化嵌入模型
        self._init_embedding_model()
        
        # 使用并行RAG处理
        response, documents = self.rag.process_query(
            state.user_query, 
            embedding_model=self.embedding_model
        )
        
        rag_time = time.time() - start_time
        
        # 基于原有性能指标创建新指标
        new_metrics = PerformanceMetrics(
            routing_time=state.performance_metrics.routing_time,
            rag_time=rag_time,
            fine_tuned_time=state.performance_metrics.fine_tuned_time,
            merge_time=state.performance_metrics.merge_time,
            total_time=state.performance_metrics.total_time + rag_time
        )
        
        return {
            "rag_response": response,
            "retrieved_documents": documents,
            "performance_metrics": new_metrics
        }
    
    def _process_fine_tuned(self, state: DualPathState) -> Dict[str, Any]:
        """处理微调路径"""
        start_time = time.time()
        
        # 这里使用聊天模型作为微调路径的模拟
        try:
            chat_model = ChatOpenAI(
                model="deepseek-ai/DeepSeek-V3",
                openai_api_key=os.environ["SILICONFLOW_API_KEY"],
                openai_api_base="https://api.siliconflow.cn/v1",
                temperature=0.7,
                timeout=30
            )
            
            response = chat_model.invoke([
                {"role": "system", "content": "你是一个专业的AI助手，提供详细和准确的回答。"},
                {"role": "user", "content": state.user_query}
            ])
            
            fine_tuned_time = time.time() - start_time
            
            # 基于原有性能指标创建新指标
            new_metrics = PerformanceMetrics(
                routing_time=state.performance_metrics.routing_time,
                rag_time=state.performance_metrics.rag_time,
                fine_tuned_time=fine_tuned_time,
                merge_time=state.performance_metrics.merge_time,
                total_time=state.performance_metrics.total_time + fine_tuned_time
            )
            
            return {
                "fine_tuned_response": response.content,
                "performance_metrics": new_metrics
            }
        except Exception as e:
            print(f"微调路径错误: {e}")
            fine_tuned_time = time.time() - start_time
            
            # 基于原有性能指标创建新指标
            new_metrics = PerformanceMetrics(
                routing_time=state.performance_metrics.routing_time,
                rag_time=state.performance_metrics.rag_time,
                fine_tuned_time=fine_tuned_time,
                merge_time=state.performance_metrics.merge_time,
                total_time=state.performance_metrics.total_time + fine_tuned_time
            )
            
            return {
                "fine_tuned_response": f"错误: {str(e)}",
                "performance_metrics": new_metrics
            }
    
    def _merge_responses(self, state: DualPathState) -> Dict[str, Any]:
        """合并响应"""
        start_time = time.time()
        
        if state.routing_decision == "rag":
            final_response = state.rag_response
        else:
            final_response = state.fine_tuned_response
        
        merge_time = time.time() - start_time
        
        # 基于原有性能指标创建新指标
        new_metrics = PerformanceMetrics(
            routing_time=state.performance_metrics.routing_time,
            rag_time=state.performance_metrics.rag_time,
            fine_tuned_time=state.performance_metrics.fine_tuned_time,
            merge_time=merge_time,
            total_time=state.performance_metrics.total_time + merge_time
        )
        
        return {
            "final_response": final_response,
            "performance_metrics": new_metrics
        }
    
    def run(self, user_query: str) -> Dict[str, Any]:
        """运行智能路由"""
        initial_state = self.state_type(user_query=user_query)
        result = self.compiled_graph.invoke(initial_state)
        
        # 处理返回结果（可能是字典）
        if isinstance(result, dict):
            # 记录性能数据
            performance_metrics = result.get('performance_metrics')
            if isinstance(performance_metrics, PerformanceMetrics):
                self.performance_monitor.record_metrics(performance_metrics)
            else:
                # 转换为 PerformanceMetrics 对象
                performance_metrics = PerformanceMetrics(**(performance_metrics or {}))
                self.performance_monitor.record_metrics(performance_metrics)
            
            return {
                "response": result.get('final_response'),
                "routing_decision": result.get('routing_decision'),
                "performance": performance_metrics.dict(),
                "retrieved_documents": result.get('retrieved_documents', [])
            }
        else:
            # 记录性能数据
            self.performance_monitor.record_metrics(result.performance_metrics)
            
            return {
                "response": result.final_response,
                "routing_decision": result.routing_decision,
                "performance": result.performance_metrics.dict(),
                "retrieved_documents": result.retrieved_documents
            }
    
    def shutdown(self):
        """关闭资源"""
        if hasattr(self, 'rag'):
            self.rag.shutdown()
