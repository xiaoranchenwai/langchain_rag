import bs4
from langchain.agents import AgentState, create_agent
from langchain_community.document_loaders import WebBaseLoader
from langchain.messages import MessageLikeRepresentation
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain.tools import tool
from langchain.messages import MessageLikeRepresentation
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ParentDocumentRetriever

import os
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain.tools import tool
#from langgraph.store.memory import InMemoryStore
from langchain_core.stores import InMemoryStore
from langchain_chroma import Chroma
import pickle
from langchain_core.messages import ToolMessage
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever


from langchain.agents.middleware import dynamic_prompt, ModelRequest


model = ChatOpenAI(model="qwen3-32b",base_url='http://10.250.2.26:8004/v1',api_key='none')

from langchain_core.vectorstores import InMemoryVectorStore

embeddings = OpenAIEmbeddings(
    model="nlp_gte_sentence-embedding_chinese-base",
    api_key="none",  # 如果需要
    base_url="http://10.250.2.23:9997/v1"
)
# 创建 ParentDocumentRetriever
# 父文档的存储层
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
# 用于创建父文档的文本分割器
#parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
# 用于创建子文档的文本分割器
# 它应该创建比父文档小的文档
child_splitter = RecursiveCharacterTextSplitter(chunk_size=20,chunk_overlap=10)

persist_dir="/home/user/cx/new_data/rag/chroma_langchain_db"
docstore_path = "/home/user/cx/new_data/rag/docstore.pkl"
# 创建或加载 Chroma 向量存储
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory=persist_dir,
)
#vector_store.delete_collection()
# 检查集合是否已有数据
collection = vector_store._collection
existing_count = collection.count()

# 只在第一次或集合为空时加载文档
if existing_count == 0:
    print("首次运行,正在加载文档...")
    path = "/home/user/cx/new_data/rag/word/"
    loader = DirectoryLoader(path, glob="**/*.docx")
    docs = loader.load()
    
    docstore = InMemoryStore()
    
    parent_retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": 3},
    )
    # 分批添加文档
    batch_size = 1  # 每批处理100个文档
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        parent_retriever.add_documents(batch)
        print(f"已处理 {min(i + batch_size, len(docs))}/{len(docs)} 个文档")
    #retriever.add_documents(docs)
    print(f"已添加 {len(docs)} 个文档")
    # 保存 docstore
    with open(docstore_path, 'wb') as f:
        pickle.dump(docstore.store, f)
    print("文档添加完成,docstore 已保存")
    
    # 2. 创建 BM25Retriever (关键词检索)
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 3

    # 3. 使用 EnsembleRetriever 组合两者
    retriever = EnsembleRetriever(
        retrievers=[parent_retriever, bm25_retriever],
        weights=[0.5, 0.5],  # 调整权重,0.5表示各占50%
    )
else:
    print(f"集合已存在,包含 {existing_count} 个文档")
    path = "/home/user/cx/agents/rag/word/"
    loader = DirectoryLoader(path, glob="**/*.docx")
    docs = loader.load()
    docstore = InMemoryStore()
    if os.path.exists(docstore_path):
        with open(docstore_path, 'rb') as f:
            docstore.store = pickle.load(f)
        print("docstore 已加载")
    else:
        print("警告: docstore 文件不存在")
    parent_retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": 3},
    )
    
    # 2. 创建 BM25Retriever (关键词检索)
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 3

    # 3. 使用 EnsembleRetriever 组合两者
    retriever = EnsembleRetriever(
        retrievers=[parent_retriever, bm25_retriever],
        weights=[0.5, 0.5],  # 调整权重,0.5表示各占50%
    )
    

@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Inject context into state messages."""
    last_query = request.state["messages"][-1].text
    retrieved_docs = retriever.invoke(last_query)

    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    system_message = (
f'''你是一个智能问答助手。你将收到一些检索到的相关文档片段，请仅基于这些文档内容回答用户的问题。

检索到的文档：
{docs_content}

重要规则：
1. 只使用上述文档中的信息回答问题
2. 如果文档中没有相关信息，明确告知用户"抱歉，提供的文档中没有找到相关信息"
3. 不要编造或推测文档中没有的内容
4. 保持回答简洁准确
    ''')

    return system_message


agent = create_agent(model, tools=[], middleware=[prompt_with_context])

query = "城市道路养护的人员要求有些？"


result = agent.invoke(
    {"messages": [{"role": "user", "content": query}]}
)
# 分离不同类型的消息
tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]

if tool_messages:
    print(f"检索到 {len(tool_messages)} 个上下文:")
    for i, msg in enumerate(tool_messages, 1):
        print(f"\n上下文 {i}: {msg.content}")
else:
    print("没有使用检索工具")

# 最终回答
print(f"\n最终回答: {result['messages'][-1].content}")