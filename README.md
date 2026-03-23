# DualPathRag - 智能路由双线架构

## 项目概述

DualPathRag 是一个智能路由双线架构系统，结合了 RAG (Retrieval-Augmented Generation) 和微调模型的优势，根据查询的复杂度自动选择最优路径，提供更准确、更快速的AI回答。

## 核心特性

- **智能路由**：基于查询复杂度自动选择 RAG 或微调模型路径
- **并行处理**：使用多线程并行检索文档
- **性能监控**：实时记录响应时间和其他性能指标
- **多文档支持**：支持上传和检索多个文档
- **中文优化**：针对中文查询和文档进行了优化

## 技术栈

- Python 3.13+
- LangChain
- Chroma 向量数据库
- OpenAI API (通过硅基流动)
- LangGraph

## 安装指南

### 1. 克隆项目

```bash
git clone <repository-url>
cd DualPathRag
```

### 2. 安装依赖

```bash
pip install -r dual_path_rag/requirements.txt
```

### 3. 设置 API Key

需要设置硅基流动 (SiliconFlow) 的 API Key：

**Windows PowerShell**:
```powershell
$env:SILICONFLOW_API_KEY = "your-api-key"
```

**Windows CMD**:
```cmd
set SILICONFLOW_API_KEY=your-api-key
```

**Linux/Mac**:
```bash
export SILICONFLOW_API_KEY=your-api-key
```

### 4. 上传文档

创建 `upload_documents.py` 文件：

```python
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
    
    embedding_model = OpenAIEmbeddings(
        model="BAAI/bge-m3",
        openai_api_key=api_key,
        openai_api_base="https://api.siliconflow.cn/v1"
    )
    
    rag = ParallelRAG()
    file_path = "your_document.txt"
    
    print(f"正在上传文档: {file_path}")
    success = rag.upload_document(file_path, embedding_model)
    
    rag.shutdown()
    
    if success:
        print("文档上传成功！")
        return True
    else:
        print("文档上传失败")
        return False

if __name__ == "__main__":
    main()
```

运行上传脚本：

```bash
python upload_documents.py
```

## 快速开始

### 运行主程序

```bash
python dual_path_rag/run.py
```

### 交互示例

```
智能路由双线架构 - 运行
============================================================
初始化智能路由双线架构...

系统已就绪，输入 'exit' 退出
============================================================

用户: 什么是人工智能？

AI: 思考中...

AI: 人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能才能完成的任务的系统。这些任务包括学习、推理、问题解决、感知、语言理解和决策制定。

路由决策: rag
响应时间: 6.89 秒
性能指标: {'routing_time': 0.49, 'rag_time': 6.41, 'fine_tuned_time': 0.0, 'merge_time': 0.0, 'total_time': 6.89, 'accuracy_score': None, 'relevance_score': None}

检索到的文档:
  1. 人工智能的定义与基础

人工智能的定义与基础

人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能才能完成的任务的系统。这些任务包括学习、推理、问题解决、感知、语言理解和决策制定。人工智能的定义可以从多个层面来理解。

第一个层面，人工智能是指智能地把某件特定的事情做好，在某个领域增强人类的智慧，这种方式又叫做智能增强。例如，搜索引擎通过智能算法为用户提供最相关的搜索结果，推荐系统根据用户的兴趣推荐内容，这些都是人工智能在特定领域的应用。

第二个层面，人工智能是指像人类一样能认知、思考、判断，模拟人类的智能。这是更高级的人工智能，目标是创造出具有人类水平智能的系统。这种类型的人工智能包括通用人工智能（AGI），能够在各种任务中表现出与人类相当的智能水平。

人工智能的发展历史可以追溯到20世纪50年代。1956年，约翰·麦卡锡在达特茅斯会议上首次提出了"人工智能"这一术语，标志着人工智能作为一个独立学科的诞生。从那时起，人工智能经历了多次起伏，包括符号主义、连接主义和行为主义等不同的发展阶段。

人工智能的核心技术包括机器学习、深度学习、自然语言处理、计算机视觉、知识表示和推理等。这些技术相互配合，使得人工智能系统能够处理各种复杂的任务。

机器学习是人工智能的一个重要分支，它使计算机系统能够从数据中学习，而不是通过明确的编程指令。机器学习算法可以分为监督学习、无监督学习和强化学习三大类。监督学习使用标记的数据进行训练，无监督学习从未标记的数据中发现模式，强化学习通过与环境交互来学习最优策略。

深度学习是机器学习的一个子领域，它基于人工神经网络，特别是深度神经网络。深度学习在图像识别、语音识别、自然语言处理等领域取得了突破性的进展。深度学习的成功得益于大数据的可用性、计算能力的提升以及算法的改进。
...
```

## 路由机制

系统根据查询的复杂度自动选择路由路径：

| 复杂度分数 | 路由决策 | 适用场景 |
|-----------|---------|----------|
| ≤ 0.4 | `rag` | 简单事实性问题，基于文档的准确回答 |
| > 0.4 | `fine_tuned` | 复杂推理问题，需要深度理解的问题 |

### 复杂度计算因素

1. **查询长度**：长查询复杂度更高
2. **关键词**：包含复杂关键词（如"如何"、"为什么"）增加复杂度
3. **语法结构**：包含逻辑词和问号增加复杂度
4. **专业术语**：包含技术术语增加复杂度

## 性能指标

系统会记录以下性能指标：

| 指标 | 含义 |
|-----|------|
| `routing_time` | 路由决策时间 |
| `rag_time` | RAG路径处理时间 |
| `fine_tuned_time` | 微调模型路径处理时间 |
| `merge_time` | 响应合并时间 |
| `total_time` | 总响应时间 |
| `accuracy_score` | 回答精确性评分（未实现） |
| `relevance_score` | 回答相关性评分（未实现） |

## 项目结构

```
DualPathRag/
├── dual_path_rag/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── dual_path_graph.py    # 双线架构核心
│   │   ├── parallel_rag.py       # 并行RAG处理
│   │   ├── performance_monitor.py # 性能监控
│   │   ├── smart_routing.py      # 智能路由引擎
│   ├── requirements.txt          # 依赖项
│   ├── run.py                    # 主程序
│   ├── test_safe.py              # 安全测试
├── vector_db/                    # 向量数据库
│   └── chroma.sqlite3            # Chroma数据库文件
├── test_document.txt             # 测试文档
└── README.md                     # 项目说明
```

## 依赖项

```
langchain
langchain-openai
langchain-community
langchain-chroma
chromadb
pydantic
nltk
```

## 故障排除

### 常见问题

1. **API Key 错误**
   - 确保正确设置了 `SILICONFLOW_API_KEY` 环境变量
   - 检查API Key是否有效

2. **文档上传失败**
   - 确保文件存在且为UTF-8编码
   - 检查文件路径是否正确

3. **响应时间过长**
   - 检查网络连接
   - 考虑使用本地模型以减少API调用时间

4. **RAG返回"没有找到相关信息"**
   - 确保已上传相关文档
   - 检查文档内容是否包含与查询相关的信息

### 调试

运行测试脚本检查系统状态：

```bash
python dual_path_rag/test_safe.py
```

## 未来规划

- [ ] 实现精确性和相关性评分
- [ ] 添加本地模型支持
- [ ] 优化响应速度
- [ ] 添加Web界面
- [ ] 支持更多文档格式

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 许可证

MIT License
