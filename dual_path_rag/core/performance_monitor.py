from pydantic import BaseModel
from typing import List, Optional
import time


class PerformanceMetrics(BaseModel):
    """性能指标"""
    routing_time: float = 0.0
    rag_time: float = 0.0
    fine_tuned_time: float = 0.0
    merge_time: float = 0.0
    total_time: float = 0.0
    accuracy_score: Optional[float] = None
    relevance_score: Optional[float] = None
    
    def dict(self) -> dict:
        """转换为字典"""
        return {
            "routing_time": self.routing_time,
            "rag_time": self.rag_time,
            "fine_tuned_time": self.fine_tuned_time,
            "merge_time": self.merge_time,
            "total_time": self.total_time,
            "accuracy_score": self.accuracy_score,
            "relevance_score": self.relevance_score
        }


class RealtimeMonitor:
    """实时监控"""
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics = {}
    
    def start(self, stage: str):
        """开始监控"""
        self.metrics[stage] = {"start": time.time()}
    
    def end(self, stage: str):
        """结束监控"""
        if stage in self.metrics:
            self.metrics[stage]["end"] = time.time()
            self.metrics[stage]["duration"] = self.metrics[stage]["end"] - self.metrics[stage]["start"]
    
    def get_metrics(self) -> dict:
        """获取指标"""
        return self.metrics
    
    def get_total_time(self) -> float:
        """获取总时间"""
        return time.time() - self.start_time


class PerformanceMonitor:
    """性能监控"""
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.realtime_monitor = RealtimeMonitor()
    
    def record_metrics(self, metrics: PerformanceMetrics):
        """记录指标"""
        self.metrics_history.append(metrics)
    
    def get_average_metrics(self) -> PerformanceMetrics:
        """获取平均指标"""
        if not self.metrics_history:
            return PerformanceMetrics()
        
        total_metrics = PerformanceMetrics()
        count = len(self.metrics_history)
        
        for metrics in self.metrics_history:
            total_metrics.routing_time += metrics.routing_time
            total_metrics.rag_time += metrics.rag_time
            total_metrics.fine_tuned_time += metrics.fine_tuned_time
            total_metrics.merge_time += metrics.merge_time
            total_metrics.total_time += metrics.total_time
            if metrics.accuracy_score:
                total_metrics.accuracy_score = total_metrics.accuracy_score or 0
                total_metrics.accuracy_score += metrics.accuracy_score
            if metrics.relevance_score:
                total_metrics.relevance_score = total_metrics.relevance_score or 0
                total_metrics.relevance_score += metrics.relevance_score
        
        return PerformanceMetrics(
            routing_time=total_metrics.routing_time / count,
            rag_time=total_metrics.rag_time / count,
            fine_tuned_time=total_metrics.fine_tuned_time / count,
            merge_time=total_metrics.merge_time / count,
            total_time=total_metrics.total_time / count,
            accuracy_score=total_metrics.accuracy_score / count if total_metrics.accuracy_score else None,
            relevance_score=total_metrics.relevance_score / count if total_metrics.relevance_score else None
        )
    
    def get_p95_metrics(self) -> PerformanceMetrics:
        """获取P95指标"""
        if not self.metrics_history:
            return PerformanceMetrics()
        
        # 按总时间排序
        sorted_metrics = sorted(self.metrics_history, key=lambda x: x.total_time)
        p95_index = int(len(sorted_metrics) * 0.95)
        return sorted_metrics[p95_index] if p95_index < len(sorted_metrics) else sorted_metrics[-1]
    
    def get_summary(self) -> dict:
        """获取性能摘要"""
        if not self.metrics_history:
            return {"message": "暂无性能数据"}
        
        avg_metrics = self.get_average_metrics()
        p95_metrics = self.get_p95_metrics()
        
        return {
            "total_requests": len(self.metrics_history),
            "average_metrics": avg_metrics.dict(),
            "p95_metrics": p95_metrics.dict(),
            "metrics_history": [m.dict() for m in self.metrics_history]
        }
    
    def reset(self):
        """重置监控"""
        self.metrics_history = []
        self.realtime_monitor = RealtimeMonitor()
