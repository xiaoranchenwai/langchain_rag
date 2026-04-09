import pandas as pd
import json
import re
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import numpy as np
import requests

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
from langchain_core.stores import InMemoryStore
from langchain_chroma import Chroma
import pickle
from langchain_core.messages import ToolMessage
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# ===================== 新增: 重排序模块 =====================

class Reranker:
    """重排序器：使用本地重排序服务对检索结果进行重新排序"""
    
    def __init__(self, 
                 rerank_url: str = "http://10.250.2.23:9998/v1/rerank",
                 model: str = "Qwen3-Reranker-0.6B",
                 top_n: int = 5):
        """
        Args:
            rerank_url: 重排序服务的URL
            model: 重排序模型名称
            top_n: 返回top-n个重排序后的文档
        """
        self.rerank_url = rerank_url
        self.model = model
        self.top_n = top_n
    
    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """
        对检索到的文档进行重排序
        
        Args:
            query: 用户查询
            documents: 检索到的文档列表
            
        Returns:
            重排序后的文档列表
        """
        if not documents:
            return documents
        
        try:
            # 准备文档内容列表
            doc_texts = [doc.page_content for doc in documents]
            
            # 调用重排序服务
            payload = {
                "model": self.model,
                "query": query,
                "documents": doc_texts
            }
            
            print(f"  🔄 正在重排序 {len(documents)} 个文档...")
            
            response = requests.post(
                self.rerank_url,
                headers={
                    "accept": "application/json",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"  ⚠️  重排序服务返回错误: {response.status_code}")
                return documents[:self.top_n]
            
            result = response.json()
            
            # 解析重排序结果
            if "results" not in result:
                print(f"  ⚠️  重排序返回格式错误")
                return documents[:self.top_n]
            
            # 按相关性分数排序
            ranked_results = sorted(
                result["results"],
                key=lambda x: x["relevance_score"],
                reverse=True
            )
            
            # 重新排列文档
            reranked_docs = []
            for item in ranked_results[:self.top_n]:
                idx = item["index"]
                score = item["relevance_score"]
                
                if idx < len(documents):
                    doc = documents[idx]
                    # 在metadata中添加重排序分数
                    doc.metadata["rerank_score"] = score
                    reranked_docs.append(doc)
            
            print(f"  ✅ 重排序完成，返回top-{len(reranked_docs)}个文档")
            print(f"     相关性分数范围: {reranked_docs[0].metadata['rerank_score']:.4f} ~ {reranked_docs[-1].metadata['rerank_score']:.4f}")
            
            return reranked_docs
            
        except requests.exceptions.Timeout:
            print(f"  ⚠️  重排序服务超时，返回原始结果")
            return documents[:self.top_n]
        except requests.exceptions.RequestException as e:
            print(f"  ⚠️  重排序服务请求失败: {e}")
            return documents[:self.top_n]
        except Exception as e:
            print(f"  ⚠️  重排序过程出错: {e}")
            return documents[:self.top_n]


# ===================== 新增: 查询优化模块 =====================

class QueryOptimizer:
    """查询优化器:实现查询扩展、后退提示、多查询策略"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def query_rewrite(self, query: str) -> str:
        """查询改写:使用大模型优化查询表达"""
        prompt = f"""请将以下用户查询改写为更清晰、更适合检索的形式。
保持原意,但使用更准确的术语和表达。只返回改写后的查询,不要解释。

原始查询: {query}

改写后的查询:"""
        
        try:
            response = self.llm.invoke(prompt)
            rewritten = response.content.strip()
            print(f"  📝 查询改写: {query} -> {rewritten}")
            return rewritten
        except Exception as e:
            print(f"  ⚠️  查询改写失败: {e}, 使用原查询")
            return query
    
    def step_back_prompting(self, query: str) -> str:
        """后退提示:生成更抽象的通用问题"""
        prompt = f"""给定一个具体的问题,请提出一个更抽象、更高层次的相关问题。
这个抽象问题应该涵盖原问题所属的更广泛的概念或原则。

具体问题: {query}

请直接给出抽象问题,不要解释:"""
        
        try:
            response = self.llm.invoke(prompt)
            abstract_query = response.content.strip()
            print(f"  🔙 后退问题: {abstract_query}")
            return abstract_query
        except Exception as e:
            print(f"  ⚠️  后退提示失败: {e}")
            return query
    
    def multi_query_generation(self, query: str, num_queries: int = 3) -> List[str]:
        """多查询生成:从不同角度生成查询变体"""
        prompt = f"""请从{num_queries}个不同的角度重新表述以下问题。
每个变体应该:
1. 保持原问题的核心意图
2. 使用不同的措辞和表达方式
3. 可能关注问题的不同方面

原始问题: {query}

请以JSON数组格式返回{num_queries}个查询变体,例如: ["查询1", "查询2", "查询3"]
只返回JSON数组,不要其他内容:"""
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            # 尝试解析JSON
            if content.startswith('[') and content.endswith(']'):
                queries = json.loads(content)
            else:
                # 如果不是纯JSON,尝试提取
                match = re.search(r'\[.*\]', content, re.DOTALL)
                if match:
                    queries = json.loads(match.group())
                else:
                    queries = [query]
            
            print(f"  🔄 生成{len(queries)}个查询变体:")
            for i, q in enumerate(queries, 1):
                print(f"     {i}. {q}")
            
            return queries
        except Exception as e:
            print(f"  ⚠️  多查询生成失败: {e}, 使用原查询")
            return [query]


class HypotheticalQuestionGenerator:
    """假设性问题生成器:为文档生成相关问题"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def generate_questions(self, doc_content: str, num_questions: int = 3) -> List[str]:
        """为文档内容生成假设性问题"""
        prompt = f"""请为以下文档内容生成{num_questions}个可能的问题。
这些问题应该是用户在查找这类信息时可能会问的。

文档内容:
{doc_content}...

请以JSON数组格式返回{num_questions}个问题,例如: ["问题1", "问题2", "问题3"]
只返回JSON数组:"""
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            if content.startswith('[') and content.endswith(']'):
                questions = json.loads(content)
            else:
                match = re.search(r'\[.*\]', content, re.DOTALL)
                if match:
                    questions = json.loads(match.group())
                else:
                    questions = []
            
            return questions
        except Exception as e:
            print(f"  ⚠️  问题生成失败: {e}")
            return []


class ContextExpander:
    """上下文扩展器:返回匹配块及其前后相邻块"""
    
    def __init__(self, docstore: InMemoryStore, window_size: int = 1):
        """
        Args:
            docstore: 文档存储
            window_size: 前后扩展的块数量(默认前后各1块)
        """
        self.docstore = docstore
        self.window_size = window_size
    
    def expand_context(self, matched_docs: List[Document]) -> List[Document]:
        """扩展上下文:返回匹配文档及其相邻文档"""
        expanded_docs = []
        seen_ids = set()
        
        for doc in matched_docs:
            doc_id = doc.metadata.get('doc_id', '')
            
            # 添加匹配的文档本身
            if doc_id not in seen_ids:
                expanded_docs.append(doc)
                seen_ids.add(doc_id)
            
            # 尝试获取前后相邻的文档块
            # 假设doc_id格式为 "parent_id/chunk_index"
            try:
                if '/' in doc_id:
                    parent_id, chunk_idx_str = doc_id.rsplit('/', 1)
                    chunk_idx = int(chunk_idx_str)
                    
                    # 获取前后相邻块
                    for offset in range(-self.window_size, self.window_size + 1):
                        if offset == 0:
                            continue  # 跳过当前块
                        
                        neighbor_id = f"{parent_id}/{chunk_idx + offset}"
                        
                        if neighbor_id not in seen_ids:
                            # 尝试从docstore获取
                            neighbor_doc = self.docstore.mget([neighbor_id])
                            if neighbor_doc and neighbor_doc[0]:
                                expanded_docs.append(Document(
                                    page_content=neighbor_doc[0].page_content,
                                    metadata={**neighbor_doc[0].metadata, 'expanded': True}
                                ))
                                seen_ids.add(neighbor_id)
            except Exception as e:
                print(f"  ⚠️  扩展上下文失败: {e}")
                continue
        
        print(f"  📖 上下文扩展: {len(matched_docs)} -> {len(expanded_docs)} 个文档块")
        return expanded_docs


# ===================== 优化的Agent配置 =====================

def setup_enhanced_rag_agent(
    enable_query_rewrite: bool = True,
    enable_step_back: bool = True,
    enable_multi_query: bool = True,
    enable_context_expansion: bool = True,
    enable_hypothetical_questions: bool = False,
    enable_reranking: bool = True,
    num_multi_queries: int = 3,
    context_window_size: int = 1,
    rerank_top_n: int = 5,
    rerank_url: str = "http://10.250.2.23:9998/v1/rerank",
    rerank_model: str = "Qwen3-Reranker-0.6B"
):
    """设置增强的RAG agent"""
    
    # 初始化模型
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

    # 初始化查询优化器
    query_optimizer = QueryOptimizer(model)
    
    # 初始化重排序器
    reranker = Reranker(
        rerank_url=rerank_url,
        model=rerank_model,
        top_n=rerank_top_n
    ) if enable_reranking else None
    
    # 父文档分割器(较大块,用于最终返回上下文)
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        add_start_index=True,
    )

    # 子文档分割器(较小块,用于精确匹配)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=20,  # 更小的块以提高匹配精度
        chunk_overlap=5
    )

    persist_dir = "/home/user/cx/agents/rag/chroma_langchain_db_enhanced"
    docstore_path = "/home/user/cx/agents/rag/enhanced_docstore_all.pkl"
    hypothetical_questions_path = "/home/user/cx/agents/rag/hypothetical_questions_all.pkl"

    # 创建或加载 Chroma 向量存储
    vector_store = Chroma(
        collection_name="enhanced_rag_collection_all",
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    collection = vector_store._collection
    existing_count = collection.count()

    # 加载或创建文档
    if existing_count == 0:
        print("🔧 首次运行,正在加载文档...")
        path = "/home/user/cx/new_data/rag/word/"
        loader = DirectoryLoader(path, glob="**/*.docx")
        docs = loader.load()
        
        docstore = InMemoryStore()
        
        parent_retriever = ParentDocumentRetriever(
            vectorstore=vector_store,
            docstore=docstore,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
            search_kwargs={"k": 3},  # 增加检索数量
        )
        
        # 分批添加文档
        batch_size = 1
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            parent_retriever.add_documents(batch)
            print(f"  已处理 {min(i + batch_size, len(docs))}/{len(docs)} 个文档")
        
        # 保存 docstore
        with open(docstore_path, 'wb') as f:
            pickle.dump(docstore.store, f)
        
        # 生成假设性问题(如果启用)
        if enable_hypothetical_questions:
            print("🤔 正在生成假设性问题...")
            hyp_gen = HypotheticalQuestionGenerator(model)
            hypothetical_data = []
            
            for i, doc in enumerate(docs):
                questions = hyp_gen.generate_questions(doc.page_content, num_questions=3)
                for q in questions:
                    hypothetical_data.append({
                        'question': q,
                        'doc_id': doc.metadata.get('source', f'doc_{i}'),
                        'doc_content': doc.page_content
                    })
                
                if (i + 1) % 10 == 0:
                    print(f"  已为 {i + 1}/{len(docs)} 个文档生成问题")
            
            # 将假设性问题也加入向量库
            hyp_docs = [
                Document(
                    page_content=item['question'],
                    metadata={'type': 'hypothetical', 'source': item['doc_id']}
                )
                for item in hypothetical_data
            ]
            vector_store.add_documents(hyp_docs)
            
            # 保存假设性问题映射
            with open(hypothetical_questions_path, 'wb') as f:
                pickle.dump(hypothetical_data, f)
            
            print(f"  ✅ 生成了 {len(hypothetical_data)} 个假设性问题")
        
        print("✅ 文档加载完成")
        
        # BM25检索器
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = 3
        
    else:
        print(f"✅ 集合已存在,包含 {existing_count} 个文档")
        docstore = InMemoryStore()
        
        if os.path.exists(docstore_path):
            with open(docstore_path, 'rb') as f:
                docstore.store = pickle.load(f)
            print("✅ docstore 已加载")
        
        # 重建检索器
        parent_retriever = ParentDocumentRetriever(
            vectorstore=vector_store,
            docstore=docstore,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
            search_kwargs={"k": 3},
        )
        
        # 加载原始文档用于BM25
        path = "/home/user/cx/new_data/rag/word/"
        loader = DirectoryLoader(path, glob="**/*.docx")
        docs = loader.load()
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = 3
    
    # 组合检索器
    ensemble_retriever = EnsembleRetriever(
        retrievers=[parent_retriever, bm25_retriever],
        weights=[0.5, 0.5],
    )
    
    # 上下文扩展器
    context_expander = ContextExpander(docstore, window_size=context_window_size) if enable_context_expansion else None
    
    # 增强的检索函数
    def enhanced_retrieve(query: str) -> List[Document]:
        """使用多种策略进行增强检索"""
        all_docs = []
        seen_content = set()
        
        queries_to_search = [query]
        
        print(f"\n🔍 开始增强检索...")
        print(f"  原始查询: {query}")
        
        # 1. 查询改写
        if enable_query_rewrite:
            rewritten = query_optimizer.query_rewrite(query)
            if rewritten != query:
                queries_to_search.append(rewritten)
        
        # 2. 后退提示
        if enable_step_back:
            abstract_query = query_optimizer.step_back_prompting(query)
            if abstract_query != query:
                queries_to_search.append(abstract_query)
        
        # 3. 多查询策略
        if enable_multi_query:
            multi_queries = query_optimizer.multi_query_generation(query, num_queries=num_multi_queries)
            queries_to_search.extend(multi_queries)
        
        # 去重查询
        queries_to_search = list(dict.fromkeys(queries_to_search))
        print(f"  💡 总共 {len(queries_to_search)} 个查询变体")
        
        # 对每个查询执行检索
        for q in queries_to_search:
            try:
                docs = ensemble_retriever.invoke(q)
                for doc in docs:
                    content_hash = hash(doc.page_content)
                    if content_hash not in seen_content:
                        all_docs.append(doc)
                        seen_content.add(content_hash)
            except Exception as e:
                print(f"  ⚠️  检索失败 ({q}): {e}")
        
        print(f"  📚 检索到 {len(all_docs)} 个唯一文档块")
        
        # 4. 上下文扩展
        if enable_context_expansion and context_expander:
            all_docs = context_expander.expand_context(all_docs)
        
        # 5. 重排序（最关键的一步）
        if enable_reranking and reranker and all_docs:
            all_docs = reranker.rerank(query, all_docs)
        else:
            # 如果不使用重排序，限制返回数量
            all_docs = all_docs[:6]
        
        return all_docs
    
    # Construct a tool for retrieving context
    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve information to help answer a query."""
        retrieved_docs = enhanced_retrieve(query)
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
        "必须使用retrieve_context工具来帮助解答用户疑问。"
    )
    agent = create_agent(model, tools, system_prompt=prompt)
    
    # 返回agent和检索函数
    return agent, enhanced_retrieve


# ===================== 其他辅助函数保持不变 =====================

def extract_filename_from_path(file_path: str) -> str:
    """从完整路径中提取文件名"""
    match = re.search(r'([^/\\]+)$', file_path)
    if match:
        filename = match.group(1)
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

def run_enhanced_rag_query(query: str, agent, retriever_func) -> Dict[str, Any]:
    """运行增强的RAG查询"""
    try:
        # 先执行检索
        #retrieved_docs = retriever_func(query)
        
        # 执行agent
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
            'retrieved_sources': list(set(retrieved_sources)),
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


def process_excel_with_enhanced_rag(
    input_excel_path: str,
    output_excel_path: str,
    agent,
    retriever_func,
    question_col: str = '问题',
    reference_source_col: str = '文件名',
    sheet_name: int = 0
):
    """处理Excel文件"""
    print(f"\n📖 正在读取Excel: {input_excel_path}")
    df = pd.read_excel(input_excel_path, sheet_name=sheet_name)
    df = df.fillna(method='ffill')
    
    # 初始化新列
    df['RAG答案'] = ""
    df['检索上下文数量'] = 0
    df['检索来源列表'] = ""
    df['来源匹配'] = 0
    df['重排序分数'] = ""  # 新增：记录重排序分数
    df['检索上下文详情'] = ""
    
    total_rows = len(df)
    print(f"📊 共 {total_rows} 个问题需要处理\n")
    
    for idx, row in df.iterrows():
        print(f"\n{'='*70}")
        print(f"📝 处理第 {idx + 1}/{total_rows} 个问题")
        print(f"{'='*70}")
        
        question = row[question_col]
        print(f"❓ 问题: {question}")
        
        reference_source = row.get(reference_source_col, None)
        if reference_source:
            print(f"📄 参考来源: {reference_source}")
        
        try:
            rag_result = run_enhanced_rag_query(question, agent, retriever_func)
            
            if rag_result['success']:
                df.at[idx, 'RAG答案'] = rag_result['final_answer']
                df.at[idx, '检索上下文数量'] = rag_result['num_contexts']
                df.at[idx, '检索来源列表'] = " | ".join(rag_result['retrieved_sources'])
                
                # 将检索上下文转换为可读格式
                context_details = []
                for i, ctx in enumerate(rag_result['retrieved_contexts'], 1):
                    # 限制内容长度避免Excel单元格过大
                    content_preview = ctx['content']
                    context_details.append(
                        f"【上下文{i}】\n"
                        f"{content_preview}\n"
                    )
                df.at[idx, '检索上下文详情'] = "\n".join(context_details)
                # # 提取重排序分数
                # rerank_scores = []
                # for ctx in rag_result['retrieved_contexts']:
                #     if 'rerank_score' in ctx:
                #         rerank_scores.append(f"{ctx['rerank_score']:.4f}")
                
                # if rerank_scores:
                #     df.at[idx, '重排序分数'] = " | ".join(rerank_scores)
                
                # 检查来源匹配
                source_match = 1 if reference_source in rag_result['retrieved_sources'] else 0
                df.at[idx, '来源匹配'] = source_match
                
                print(f"\n✅ 检索成功")
                print(f"  📚 检索到 {rag_result['num_contexts']} 个上下文")
                print(f"  📂 来源: {', '.join(rag_result['retrieved_sources'][:3])}")
                # if rerank_scores:
                #     print(f"  🎯 重排序分数: {', '.join(rerank_scores[:3])}")
                print(f"  {'✓' if source_match else '✗'} 来源匹配: {'是' if source_match else '否'}")
            else:
                df.at[idx, 'RAG答案'] = rag_result['final_answer']
                print(f"\n❌ 查询失败: {rag_result.get('error')}")
                
        except Exception as e:
            print(f"\n❌ 处理出错: {str(e)}")
            df.at[idx, 'RAG答案'] = f"错误: {str(e)}"
    
    # 保存结果
    print(f"\n{'='*70}")
    print(f"💾 保存结果到: {output_excel_path}")
    df.to_excel(output_excel_path, index=False, engine='openpyxl')
    
    # 统计
    print(f"\n{'='*70}")
    print("📊 处理完成!统计信息:")
    print(f"{'='*70}")
    print(f"总问题数: {total_rows}")
    print(f"来源匹配成功: {df['来源匹配'].sum()} ({df['来源匹配'].sum()/total_rows*100:.2f}%)")
    print(f"平均检索上下文数: {df['检索上下文数量'].mean():.2f}")
    print(f"{'='*70}\n")
    
    return df


# ===================== 主程序 =====================

if __name__ == "__main__":
    INPUT_EXCEL = "/home/user/cx/new_data/rag/知识库问答_标注.xlsx"
    OUTPUT_EXCEL = "/home/user/cx/agents/rag/enhanced__agentic_rag_results_rerank.xlsx"
    
    print("\n" + "="*70)
    print("🚀 增强RAG系统")
    print("="*70)
    print("✨ 功能:")
    print("  1️⃣  查询改写")
    print("  2️⃣  后退提示")
    print("  3️⃣  多查询策略")
    print("  4️⃣  上下文扩展")
    print("  5️⃣  假设性问题(可选)")
    print("  6️⃣  重排序优化")
    print("="*70 + "\n")
    
    # 初始化增强RAG系统
    print("🔧 正在初始化增强RAG Agent...")
    agent, retriever_func = setup_enhanced_rag_agent(
        enable_query_rewrite=True,
        enable_step_back=True,
        enable_multi_query=True,
        enable_context_expansion=False,
        enable_hypothetical_questions=False,  # 首次运行可设为True生成问题
        enable_reranking=True,  # 启用重排序
        num_multi_queries=3,
        context_window_size=1,
        rerank_top_n=6,  # 重排序后返回top-5
        rerank_url="http://10.250.2.23:9998/v1/rerank",
        rerank_model="Qwen3-Reranker-0.6B"
    )
    print("✅ Agent初始化完成\n")
    
    # 处理Excel
    result_df = process_excel_with_enhanced_rag(
        input_excel_path=INPUT_EXCEL,
        output_excel_path=OUTPUT_EXCEL,
        agent=agent,
        retriever_func=retriever_func,
        question_col='问题',
        reference_source_col='文件名'
    )
    
    print("🎉 所有任务完成!")