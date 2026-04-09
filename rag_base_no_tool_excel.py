import pandas as pd
import json
import re
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any

# LangChain imports
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
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# from langchain_community.retrievers import ContextualCompressionRetriever
# from langchain_community.retrievers.document_compressors import LLMListwiseRerank
# from langchain_classic.retrievers.contextual_compression import (
#     ContextualCompressionRetriever,
# )
# from langchain_cohere import ChatCohere, CohereRerank

# ===================== Agent配置部分 =====================
def setup_rag_agent():
    """设置RAG agent"""
    model = ChatOpenAI(
        model="qwen3-235b",
        base_url='http://10.250.2.25:8004/v1',
        api_key='none'
    )

    embeddings = OpenAIEmbeddings(
        model="nlp_gte_sentence-embedding_chinese-base",
        api_key="none",
        base_url="http://10.250.2.23:9997/v1"
    )

    # 父文档的存储层
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )

    # 子文档的文本分割器
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=20,
        chunk_overlap=5
    )

    persist_dir = "/home/user/cx/agents/rag/chroma_langchain_db"
    docstore_path = "/home/user/cx/agents/rag/p_1000_200_c_20_5_docstore_mix_7_3_3.pkl"

    # 创建或加载 Chroma 向量存储
    vector_store = Chroma(
        collection_name="p_1000_200_c_20_5_collection_mix_7_3_3",
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

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
        
        parent_retriever = ParentDocumentRetriever(
            vectorstore=vector_store,
            docstore=docstore,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
            search_kwargs={"k": 3},
        )
        
        # 分批添加文档
        batch_size = 1
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            parent_retriever.add_documents(batch)
            print(f"已处理 {min(i + batch_size, len(docs))}/{len(docs)} 个文档")
        
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
            weights=[0.7, 0.3],  # 调整权重,0.5表示各占50%
        )
        
    else:
        print(f"集合已存在,包含 {existing_count} 个文档,跳过加载")
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
            weights=[0.7, 0.3],  # 调整权重,0.5表示各占50%
        )
    
    @dynamic_prompt
    def prompt_with_context(request: ModelRequest) -> str:
        """Inject context into state messages."""
        last_query = request.state["messages"][-1].text
        retrieved_docs = retriever.invoke(last_query)
        if len(retrieved_docs)>3:
            retrieved_docs=retrieved_docs[0:3]
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
    
    return agent

# ===================== 辅助函数部分 =====================

def extract_filename_from_path(file_path: str) -> str:
    """从完整路径中提取文件名（去除扩展名）"""
    # 提取最后的文件名部分
    match = re.search(r'([^/\\]+)$', file_path)
    if match:
        filename = match.group(1)
        # 去除.docx扩展名
        if filename.endswith('.docx'):
            return filename[:-5]
        return filename
    return file_path

def parse_tool_message_content(content: str) -> List[Dict[str, Any]]:
    """解析工具消息内容，提取source和content信息"""
    results = []
    
    # 使用正则表达式匹配Source和Content
    # 修改模式以更好地匹配你的格式
    pattern = r"Source:\s*(\{[^}]+\})\s*\nContent:\s*(.+?)(?=\n\nSource:|\Z)"
    matches = re.finditer(pattern, content, re.DOTALL)
    
    for match in matches:
        try:
            # 将字符串转换为字典
            source_str = match.group(1)
            source_dict = eval(source_str)
            
            # 提取source路径
            source_path = source_dict.get('source', '')
            filename = extract_filename_from_path(source_path)
            
            results.append({
                'filename': filename,
                'full_source': source_dict,
                'content': match.group(2).strip()
            })
        except Exception as e:
            print(f"解析source时出错: {e}")
            continue
    
    return results

def check_source_match(retrieved_sources: List[str], reference_source: str) -> int:
    """检查检索到的来源中是否包含参考来源"""
    if pd.isna(reference_source) or not reference_source:
        return 0
    
    # 标准化文件名进行比较
    reference_clean = str(reference_source).strip()
    
    for source in retrieved_sources:
        source_clean = str(source).strip()
        # 模糊匹配：检查是否包含关系
        if reference_clean in source_clean or source_clean in reference_clean:
            return 1
    
    return 0

def run_rag_query(query: str, agent) -> Dict[str, Any]:
    """运行RAG查询并返回结果"""
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": query}]}
        )
        
        # 分离不同类型的消息
        tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
        
        # 提取检索到的上下文
        retrieved_contexts = []
        retrieved_sources = []
        
        if tool_messages:
            for msg in tool_messages:
                parsed_results = parse_tool_message_content(msg.content)
                for item in parsed_results:
                    retrieved_contexts.append({
                        'source': item['filename'],
                        'full_source': item['full_source'],
                        'content': item['content']
                    })
                    retrieved_sources.append(item['filename'])
        
        # 获取最终回答
        final_answer = result['messages'][-1].content if result['messages'] else ""
        
        return {
            'final_answer': final_answer,
            'retrieved_contexts': retrieved_contexts,
            'retrieved_sources': retrieved_sources,
            'num_contexts': len(retrieved_contexts),
            'success': True
        }
    except Exception as e:
        return {
            'final_answer': f"查询失败: {str(e)}",
            'retrieved_contexts': [],
            'retrieved_sources': [],
            'num_contexts': 0,
            'success': False,
            'error': str(e)
        }

# ===================== 主处理函数 =====================

def process_excel_with_rag(
    input_excel_path: str,
    output_excel_path: str,
    agent,
    question_col: str = '问题',
    reference_answer_col: str = '参考答案',
    reference_source_col: str = '答案来源文档名',
    sheet_name: int = 0
):
    """
    处理Excel文件，对每个问题进行RAG检索
    
    参数:
        input_excel_path: 输入Excel文件路径
        output_excel_path: 输出Excel文件路径
        agent: RAG agent对象
        question_col: 问题列名
        reference_answer_col: 参考答案列名
        reference_source_col: 答案来源文档名列名
        sheet_name: 工作表名称或索引
    """
    
    # 读取Excel文件
    print(f"正在读取Excel文件: {input_excel_path}")
    df = pd.read_excel(input_excel_path, sheet_name=sheet_name)
    
    # 使用forward fill填充空白单元格
    df = df.fillna(method='ffill')
    
    # 初始化新列
    df['RAG答案'] = ""
    df['检索上下文数量'] = 0
    df['检索上下文详情'] = ""
    df['检索来源列表'] = ""
    df['来源匹配'] = 0
    
    total_rows = len(df)
    print(f"共有 {total_rows} 个问题需要处理\n")
    
    # 遍历每一行
    for idx, row in df.iterrows():
        print(f"{'='*60}")
        print(f"处理第 {idx + 1}/{total_rows} 个问题")
        
        question = row[question_col]
        print(f"问题: {question}")
        
        reference_source = row[reference_source_col] if reference_source_col in df.columns else None
        if reference_source:
            print(f"参考来源: {reference_source}")
        
        try:
            # 运行RAG查询
            rag_result = run_rag_query(question, agent)
            
            if rag_result['success']:
                # 保存结果到DataFrame
                df.at[idx, 'RAG答案'] = rag_result['final_answer']
                df.at[idx, '检索上下文数量'] = rag_result['num_contexts']
                
                # 将检索上下文转换为可读格式
                context_details = []
                for i, ctx in enumerate(rag_result['retrieved_contexts'], 1):
                    # 限制内容长度避免Excel单元格过大
                    content_preview = ctx['content']
                    context_details.append(
                        f"【上下文{i}】\n"
                        f"内容: {content_preview}\n"
                    )
                df.at[idx, '检索上下文详情'] = "\n".join(context_details)
                
                # 保存检索来源列表
                df.at[idx, '检索来源列表'] = " | ".join(rag_result['retrieved_sources'])
                
                # 检查来源匹配
                source_match = check_source_match(
                    rag_result['retrieved_sources'],
                    reference_source
                )
                df.at[idx, '来源匹配'] = source_match
                
                print(f"✓ 检索到 {rag_result['num_contexts']} 个上下文")
                print(f"✓ 检索来源: {', '.join(rag_result['retrieved_sources'][:3])}{'...' if len(rag_result['retrieved_sources']) > 3 else ''}")
                print(f"✓ 来源匹配: {'是' if source_match == 1 else '否'}")
            else:
                df.at[idx, 'RAG答案'] = rag_result['final_answer']
                df.at[idx, '来源匹配'] = 0
                print(f"✗ 查询失败: {rag_result.get('error', '未知错误')}")
            
        except Exception as e:
            print(f"✗ 处理出错: {str(e)}")
            df.at[idx, 'RAG答案'] = f"错误: {str(e)}"
            df.at[idx, '来源匹配'] = 0
        
        print()  # 空行分隔
    
    # 保存结果到新的Excel文件
    print(f"{'='*60}")
    print(f"正在保存结果到: {output_excel_path}")
    df.to_excel(output_excel_path, index=False, engine='openpyxl')
    
    # 打印统计信息
    print(f"\n{'='*60}")
    print("处理完成！统计信息:")
    print(f"{'='*60}")
    print(f"总问题数: {total_rows}")
    print(f"来源匹配成功: {df['来源匹配'].sum()} ({df['来源匹配'].sum()/total_rows*100:.2f}%)")
    print(f"平均检索上下文数: {df['检索上下文数量'].mean():.2f}")
    print(f"最大检索上下文数: {df['检索上下文数量'].max()}")
    print(f"最小检索上下文数: {df['检索上下文数量'].min()}")
    print(f"{'='*60}")
    
    return df

# ===================== 主程序 =====================

if __name__ == "__main__":
    # 配置参数
    INPUT_EXCEL = "/home/user/cx/agents/rag/知识库问答_标注.xlsx"  # 输入Excel文件路径
    OUTPUT_EXCEL = "/home/user/cx/agents/rag/p_1000_200_c_20_5_questions_with_rag_results_notool_235b_mix_7_3.xlsx"  # 输出Excel文件路径
    
    print("="*60)
    print("RAG Excel批量处理工具")
    print("="*60)
    print()
    
    # 设置agent
    print("正在初始化RAG Agent...")
    agent = setup_rag_agent()
    print("✓ Agent初始化完成\n")
    
    # 处理Excel文件
    result_df = process_excel_with_rag(
        input_excel_path=INPUT_EXCEL,
        output_excel_path=OUTPUT_EXCEL,
        agent=agent,
        question_col='问题',
        reference_answer_col='答案',
        reference_source_col='文件名'
    )
    
    print("\n所有任务完成！")