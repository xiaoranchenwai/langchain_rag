"""
基础 RAG 系统实现
使用 LangChain 框架构建简单的检索增强生成系统
"""

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Tongyi  # 可替换为 OpenAI 等
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os


class BasicRAG:
    """基础 RAG 系统"""
    
    def __init__(self, model_name="dashscope/qwen-turbo"):
        """
        初始化 RAG 系统
        
        Args:
            model_name: 大模型名称
        """
        self.model_name = model_name
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        
    def load_documents(self, file_paths):
        """
        加载文档
        
        Args:
            file_paths: 文档路径列表
            
        Returns:
            documents: 加载的文档列表
        """
        documents = []
        
        for file_path in file_paths:
            if file_path.endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            else:
                print(f"不支持的文件格式: {file_path}")
                continue
                
            docs = loader.load()
            documents.extend(docs)
            
        print(f"成功加载 {len(documents)} 个文档")
        return documents
    
    def split_documents(self, documents, chunk_size=500, chunk_overlap=50):
        """
        文档分块
        
        Args:
            documents: 文档列表
            chunk_size: 块大小
            chunk_overlap: 块重叠大小
            
        Returns:
            chunks: 分块后的文档列表
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"文档分块完成，共 {len(chunks)} 个块")
        return chunks
    
    def build_vectorstore(self, chunks, embedding_model="BAAI/bge-small-zh-v1.5"):
        """
        构建向量数据库
        
        Args:
            chunks: 文档块列表
            embedding_model: 嵌入模型名称
        """
        print("正在构建向量数据库...")
        
        # 初始化嵌入模型
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}
        )
        
        # 创建向量数据库
        self.vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        
        print("向量数据库构建完成")
    
    def save_vectorstore(self, path="./vectorstore"):
        """保存向量数据库"""
        if self.vectorstore:
            self.vectorstore.save_local(path)
            print(f"向量数据库已保存至 {path}")
    
    def load_vectorstore(self, path="./vectorstore", embedding_model="BAAI/bge-small-zh-v1.5"):
        """加载向量数据库"""
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}
        )
        self.vectorstore = FAISS.load_local(
            path, 
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"向量数据库已从 {path} 加载")
    
    def setup_qa_chain(self, top_k=3):
        """
        设置问答链
        
        Args:
            top_k: 检索文档数量
        """
        if not self.vectorstore:
            raise ValueError("请先构建或加载向量数据库")
        
        # 创建检索器
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        )
        
        # 定义提示词模板
        prompt_template = """基于以下已知信息，简洁和专业地回答用户的问题。
如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"。
不允许在答案中添加编造成分。

已知信息：
{context}

问题：{question}

回答："""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # 初始化大模型
        llm = Tongyi(
            model_name=self.model_name,
            temperature=0.1
        )
        
        # 创建问答链
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        print("问答链设置完成")
    
    def query(self, question):
        """
        查询问题
        
        Args:
            question: 用户问题
            
        Returns:
            result: 包含答案和来源文档的字典
        """
        if not self.qa_chain:
            raise ValueError("请先设置问答链")
        
        result = self.qa_chain({"query": question})
        return result


def main():
    """主函数示例"""
    
    # 初始化 RAG 系统
    rag = BasicRAG()
    
    # 1. 离线索引构建
    print("=" * 50)
    print("步骤1: 加载文档")
    print("=" * 50)
    documents = rag.load_documents(["example.txt"])  # 替换为实际文档路径
    
    print("\n" + "=" * 50)
    print("步骤2: 文档分块")
    print("=" * 50)
    chunks = rag.split_documents(documents, chunk_size=500, chunk_overlap=50)
    
    print("\n" + "=" * 50)
    print("步骤3: 构建向量数据库")
    print("=" * 50)
    rag.build_vectorstore(chunks)
    
    # 可选：保存向量数据库
    rag.save_vectorstore("./vectorstore")
    
    # 2. 在线检索生成
    print("\n" + "=" * 50)
    print("步骤4: 设置问答链")
    print("=" * 50)
    rag.setup_qa_chain(top_k=3)
    
    print("\n" + "=" * 50)
    print("步骤5: 开始问答")
    print("=" * 50)
    
    # 测试查询
    questions = [
        "RAG系统的主要目标是什么？",
        "如何优化文档分块？"
    ]
    
    for question in questions:
        print(f"\n问题: {question}")
        result = rag.query(question)
        print(f"答案: {result['result']}")
        print(f"参考文档数: {len(result['source_documents'])}")


if __name__ == "__main__":
    main()