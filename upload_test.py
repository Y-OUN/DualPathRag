import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dual_path_rag'))

from core.parallel_rag import ParallelRAG
from langchain_openai import OpenAIEmbeddings


def main():
    api_key = os.environ.get("SILICONFLOW_API_KEY")
    if not api_key:
        print("错误: 未设置 SILICONFLOW_API_KEY 环境变量")
        return False
    
    print("初始化嵌入模型...")
    embedding_model = OpenAIEmbeddings(
        model="BAAI/bge-m3",
        openai_api_key=api_key,
        openai_api_base="https://api.siliconflow.cn/v1"
    )
    
    rag = ParallelRAG()
    
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_document.txt")
    
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在")
        return False
    
    print(f"\n正在上传文档: {file_path}")
    success = rag.upload_document(file_path, embedding_model)
    
    rag.shutdown()
    
    if success:
        print("\n文档上传成功！")
        print(f"文档已存储到: {os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vector_db'))}")
        return True
    else:
        print("\n文档上传失败")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
