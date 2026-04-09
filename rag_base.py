import bs4
from langchain.agents import AgentState, create_agent
from langchain_community.document_loaders import WebBaseLoader
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



model = ChatOpenAI(model="qwen3-32b",base_url='http://10.250.2.26:8004/v1',api_key='none')

from langchain_core.vectorstores import InMemoryVectorStore

embeddings = OpenAIEmbeddings(
    model="nlp_gte_sentence-embedding_chinese-base",
    api_key="none",  # 如果需要
    base_url="http://10.250.2.23:9997/v1"
)
test_text = "你好，世界！"

# 计算文本的嵌入
embedding = embeddings.embed_query(test_text)
print("Embedding result:", embedding)
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
child_splitter = RecursiveCharacterTextSplitter(chunk_size=50,chunk_overlap=10)

persist_dir="/home/user/cx/agents/rag/chroma_langchain_db"
docstore_path = "/home/user/cx/agents/rag/docstore.pkl"
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
    path = "/home/user/cx/agents/rag/word/"
    loader = DirectoryLoader(path, glob="**/*.docx")
    docs = loader.load()
    
    docstore = InMemoryStore()
    
    retriever = ParentDocumentRetriever(
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
        retriever.add_documents(batch)
        print(f"已处理 {min(i + batch_size, len(docs))}/{len(docs)} 个文档")
    #retriever.add_documents(docs)
    print(f"已添加 {len(docs)} 个文档")
    # 保存 docstore
    with open(docstore_path, 'wb') as f:
        pickle.dump(docstore.store, f)
    print("文档添加完成,docstore 已保存")
else:
    print(f"集合已存在,包含 {existing_count} 个文档,跳过加载")
    docstore = InMemoryStore()
    if os.path.exists(docstore_path):
        with open(docstore_path, 'rb') as f:
            docstore.store = pickle.load(f)
        print("docstore 已加载")
    else:
        print("警告: docstore 文件不存在")
    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": 3},
    )

# Construct a tool for retrieving context
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = retriever.invoke(query)
    #retrieved_docs = vector_store.similarity_search(query, k=3)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

tools = [retrieve_context]
# If desired, specify custom instructions
prompt = (
    "您可以使用一个工具来获取问题的相关信息。"
    "请使用该工具来帮助解答用户疑问。"
)
agent = create_agent(model, tools, system_prompt=prompt)


query = "城市道路养护的人员要求有些？"
# 使用 invoke 直接获取最终结果
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
# for event in agent.stream(
#     {"messages": [{"role": "user", "content": query}]},
#     stream_mode="values",
# ):
#     event["messages"][-1].pretty_print()