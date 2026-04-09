# LangChain RAG Experiments

这个仓库收集了一组基于 LangChain 的 RAG 实验脚本，重点覆盖中文知识库问答、向量索引构建、父子分块检索、BM25/向量混合检索、重排，以及面向 Excel 问答集的批量评估流程。当前代码形态更接近“研究与验证脚本集合”，而不是一个已经模块化发布的 Python 包。

## 仓库内容

- `langchain_rag.py`
  一个最基础的 RAG 示例，支持读取 `txt` / `pdf`，构建 FAISS 向量库，并通过 LangChain `RetrievalQA` 完成问答。
- `rag_base.py` / `rag_base_no_tool.py`
  面向本地知识库的代理式问答脚本，使用 Chroma 持久化向量库和父子文档检索。
- `rag_base_excel.py` / `rag_base_index.py`
  对 Excel 问题集做批量处理，用于评估检索结果、答案来源匹配率和上下文数量。
- `rag_base_index_EnsembleRetriever*.py`
  一组增强检索实验，组合了向量检索、BM25、查询改写、Agent、Rerank 和不同模型配置。

## 环境要求

- Python 3.10+
- 可访问的模型与嵌入服务
- 本地文档目录、Chroma 持久化目录、Excel 输入文件

建议先创建虚拟环境，再安装常用依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install langchain langchain-community langchain-openai langchain-chroma \
  langchain-text-splitters langchain-classic faiss-cpu pandas openpyxl \
  beautifulsoup4 pypdf sentence-transformers
```

如果要处理 `docx` 文档，通常还需要补充对应 loader 依赖。

## 快速开始

### 1. 运行基础 RAG 示例

编辑 [langchain_rag.py](langchain_rag.py)，将示例中的 `example.txt` 替换为你的文档路径，然后执行：

```bash
python langchain_rag.py
```

该脚本会依次完成：

1. 加载文档
2. 文本切分
3. 构建或保存向量库
4. 创建问答链
5. 对示例问题执行问答

### 2. 运行知识库检索/评估脚本

批处理脚本默认把输入 Excel、文档目录、Chroma 存储目录写死在文件顶部或 `__main__` 区域，例如 [rag_base_index.py](rag_base_index.py) 中的：

- `INPUT_EXCEL`
- `OUTPUT_EXCEL`
- `persist_dir`
- `docstore_path`
- `DirectoryLoader(path, glob="**/*.docx")`

修改为你本机的实际路径后再运行：

```bash
python rag_base_index.py
python rag_base_excel.py
python rag_base_index_EnsembleRetriever.py
```

这些脚本通常会输出：

- 检索到的上下文数量
- 来源文档列表
- 来源匹配统计
- 处理后的 Excel 文件

## 实验脚本选择建议

- 想看最小可运行流程：`langchain_rag.py`
- 想验证本地知识库问答：`rag_base.py`
- 想只评估检索效果：`rag_base_index.py`
- 想做混合检索对比：`rag_base_index_EnsembleRetriever.py`
- 想尝试查询优化、Agent 或 Rerank：`rag_base_index_EnsembleRetriever_opt*.py`

## 当前限制

当前脚本普遍存在以下工程化限制：

- 依赖和版本未集中管理，没有 `requirements.txt` 或 `pyproject.toml`
- 多处使用硬编码的绝对路径与内网模型地址
- 尚未抽成统一配置层，也没有自动化测试
- 不同实验脚本之间有较多重复逻辑

如果你准备长期维护这个仓库，建议优先把配置抽到环境变量或配置文件，再逐步拆出 `src/` 和 `tests/`。
