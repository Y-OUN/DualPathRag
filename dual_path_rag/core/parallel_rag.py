from typing import List, Dict, Any, Optional
import concurrent.futures
import time
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import os


class ParallelRAG:
    """并行RAG处理"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.vector_store = None
        self.embedding_model = None
    
    def upload_document(self, file_path: str, embedding_model: Optional[Any] = None):
        """上传文档"""
        if embedding_model:
            self.embedding_model = embedding_model
        
        if not self.embedding_model:
            raise ValueError("嵌入模型未初始化")
        
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                separators=["。", "！", "？", "；", "\n"],
                chunk_size=1000,
                chunk_overlap=200
            )
            
            splits = text_splitter.split_documents(documents)
            
            if not self.vector_store:
                self.vector_store = Chroma.from_documents(
                    documents=splits,
                    embedding=self.embedding_model,
                    persist_directory="./vector_db"
                )
            else:
                self.vector_store.add_documents(splits)
            
            print(f"文档上传成功，分割为 {len(splits)} 个片段")
            return True
        except Exception as e:
            print(f"文档上传失败: {e}")
            return False
    
    def process_query(self, query: str, embedding_model: Optional[Any] = None) -> tuple:
        """处理查询"""
        if embedding_model:
            self.embedding_model = embedding_model
        
        if not self.embedding_model:
            return "错误: 嵌入模型未初始化", []
        
        if not self.vector_store:
            # 创建空的向量存储
            self.vector_store = Chroma(
                embedding_function=self.embedding_model,
                persist_directory="./vector_db"
            )
        
        try:
            # 并行检索
            futures = []
            
            # 向量检索
            futures.append(self.executor.submit(
                self._vector_search, 
                query, 
                k=3
            ))
            
            # 等待结果
            results = []
            for future in concurrent.futures.as_completed(futures, timeout=10):
                try:
                    result = future.result()
                    results.extend(result)
                except concurrent.futures.TimeoutError:
                    print("检索超时")
                except Exception as e:
                    print(f"检索错误: {e}")
            
            # 生成响应
            response = self._generate_response(query, results)
            
            return response, results
        except Exception as e:
            print(f"处理查询失败: {e}")
            return f"错误: {str(e)}", []
    
    def _vector_search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """向量检索"""
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            return [{
                "content": doc.page_content,
                "score": 1.0,  # Chroma不直接返回分数
                "source": doc.metadata.get("source", "unknown")
            } for doc in docs]
        except Exception as e:
            print(f"向量检索错误: {e}")
            return []
    
    def _generate_response(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """生成响应"""
        if not documents:
            return "抱歉，没有找到相关信息。"
        
        # 构建上下文
        context = "\n".join([doc["content"] for doc in documents[:3]])
        
        # 生成提示
        prompt = f"基于以下上下文回答问题:\n\n{context}\n\n问题: {query}\n\n回答:"
        
        # 使用聊天模型生成回答
        from langchain_openai import ChatOpenAI
        
        try:
            chat_model = ChatOpenAI(
                model="deepseek-ai/DeepSeek-V3",
                openai_api_key=os.environ["SILICONFLOW_API_KEY"],
                openai_api_base="https://api.siliconflow.cn/v1",
                temperature=0.5,
                timeout=30
            )
            
            response = chat_model.invoke([
                {"role": "system", "content": "你是一个专业的AI助手，基于提供的上下文回答问题，保持回答准确、简洁。"},
                {"role": "user", "content": prompt}
            ])
            
            return response.content
        except Exception as e:
            print(f"生成响应错误: {e}")
            return f"错误: {str(e)}"
    
    def shutdown(self):
        """关闭线程池"""
        try:
            self.executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
