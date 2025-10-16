"""
RAG Engine Module
Enhanced RAG system for financial trading intelligence
Combines news, sentiment, and market data for intelligent responses
Supports both Google Gemini and OpenAI with separate vector databases
"""

import os
from typing import List, Dict, Optional

# Import for Google Gemini
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Import for OpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import chromadb

class FinancialRAGEngine:
    def __init__(self, api_key: str = None, openai_api_key: str = None, llm_provider: str = "gemini"):
        """
        Initialize the Financial RAG Engine
        
        Args:
            api_key: Google API key for Gemini
            openai_api_key: OpenAI API key
            llm_provider: "gemini" or "openai" to choose the LLM provider
        """
        self.llm_provider = llm_provider.lower()
        
        # --- API Key Setup ---
        self.google_api_key = api_key or os.getenv('GOOGLE_API_KEY')
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable or provide it.")
        elif self.llm_provider == "gemini" and not self.google_api_key:
            raise ValueError("Google API key not found. Please set GOOGLE_API_KEY environment variable or provide it.")

        
        # --- Initialize Embeddings ---
        if self.llm_provider == "openai":
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=self.openai_api_key,
                model="text-embedding-3-small"  # Faster and cheaper
            )
            print("✅ Initialized OpenAI Embeddings (text-embedding-3-small)")
            
        elif self.llm_provider == "gemini":
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.google_api_key
            )
            print("✅ Initialized Gemini Embeddings")

        else:
            raise ValueError("Invalid LLM provider specified. Choose 'gemini' or 'openai'.")
        
        # --- Initialize LLM ---
        if self.llm_provider == "gemini":
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-pro",
                google_api_key=self.google_api_key,
                temperature=0.3,
                convert_system_message_to_human=True
            )
            print("✅ Initialized Gemini LLM (gemini-pro)")
        elif self.llm_provider == "openai":
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",  # You can change to "gpt-4" for better quality
                openai_api_key=self.openai_api_key,
                temperature=0.3
            )
            print("✅ Initialized OpenAI LLM (gpt-3.5-turbo)")
        
        # Text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Vector store - SEPARATE DIRECTORIES FOR EACH PROVIDER
        self.vector_store = None
        # CRITICAL: Different persist directories for different embeddings!
        if self.llm_provider == "openai":
            self.persist_directory = "./Vector_DB_Financial_OpenAI"
        else:
            self.persist_directory = "./Vector_DB_Financial_Gemini"
        
        print(f"📁 Using vector database: {self.persist_directory}")
        
        # Conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Conversation chain
        self.conversation_chain = None
        
        # Initialize or load existing vector store
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize or load existing ChromaDB vector store"""
        try:
            # Try to load existing vector store
            if os.path.exists(self.persist_directory):
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                print(f"✅ Loaded existing vector database from {self.persist_directory}")
            else:
                # Create new vector store
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                print(f"✅ Created new vector database at {self.persist_directory}")
        except Exception as e:
            print(f"⚠️ Warning: Error with vector store: {str(e)}")
            # Fallback to in-memory if persistent fails
            self.vector_store = Chroma(
                embedding_function=self.embeddings
            )
            print("⚠️ Using in-memory vector store as fallback")
    
    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict]] = None):
        """
        Add documents to the vector store
        
        Args:
            documents: List of text documents
            metadatas: Optional list of metadata dictionaries
        """
        try:
            # Split documents into chunks
            chunks = []
            chunk_metadatas = []
            
            for i, doc in enumerate(documents):
                doc_chunks = self.text_splitter.split_text(doc)
                chunks.extend(doc_chunks)
                
                # Add metadata for each chunk
                if metadatas and i < len(metadatas):
                    chunk_metadatas.extend([metadatas[i]] * len(doc_chunks))
                else:
                    chunk_metadatas.extend([{"source": f"document_{i}"}] * len(doc_chunks))
            
            # Add to vector store
            if chunks:
                self.vector_store.add_texts(
                    texts=chunks,
                    metadatas=chunk_metadatas
                )
                
                # Persist the vector store
                if hasattr(self.vector_store, 'persist'):
                    self.vector_store.persist()
                
                print(f"✅ Added {len(chunks)} chunks to {self.llm_provider.upper()} vector database")
                return True
            return False
        except Exception as e:
            print(f"❌ Error adding documents: {str(e)}")
            return False
    
    def create_conversation_chain(self):
        """Create the conversational retrieval chain with custom prompt"""
        
        # Custom prompt template for financial advice
        template = """You are an expert financial analyst and trading advisor with deep knowledge of markets, stocks, and trading strategies.

Use the following context from recent financial news, market data, and sentiment analysis to answer the question. 
Be precise, analytical, and provide actionable insights. Always consider both opportunities and risks.

If you're unsure or the context doesn't contain enough information, say so clearly. Never make up financial data or give advice without proper context.

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Provide a comprehensive answer that includes:
1. Direct answer to the question
2. Supporting evidence from the context (cite sources)
3. Relevant technical or fundamental insights
4. Risk considerations if applicable
5. Confidence level in your analysis

Answer:"""

        PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "chat_history", "question"]
        )
        
        # Create retrieval chain
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 5}  # Retrieve top 5 relevant chunks
            ),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            return_source_documents=True,
            verbose=False
        )
        
        print(f"✅ Conversation chain created with {self.llm_provider.upper()}")
    
    def query(self, question: str, context: Optional[Dict] = None) -> Dict:
        """
        Query the RAG system
        
        Args:
            question: User's question
            context: Optional additional context (sentiment, market data, etc.)
            
        Returns:
            Dictionary with answer and sources
        """
        try:
            if not self.conversation_chain:
                self.create_conversation_chain()
            
            # Enhance question with context if provided
            enhanced_question = question
            if context:
                context_str = self._format_context(context)
                enhanced_question = f"{question}\n\nAdditional Context:\n{context_str}"
            
            # Get response
            response = self.conversation_chain({
                "question": enhanced_question
            })
            
            # Extract sources
            sources = []
            if "source_documents" in response:
                for doc in response["source_documents"]:
                    source_info = {
                        "content": doc.page_content[:200] + "...",
                        "metadata": doc.metadata
                    }
                    sources.append(source_info)
            
            return {
                "answer": response["answer"],
                "sources": sources,
                "success": True
            }
        except Exception as e:
            print(f"❌ Error querying RAG: {str(e)}")
            return {
                "answer": f"I encountered an error processing your question: {str(e)}",
                "sources": [],
                "success": False
            }
    
    def _format_context(self, context: Dict) -> str:
        """Format additional context for the query"""
        formatted = []
        
        if "sentiment" in context:
            sentiment = context["sentiment"]
            formatted.append(f"Overall Sentiment: {sentiment.get('overall_label', 'N/A')} (Score: {sentiment.get('overall_score', 0)})")
            formatted.append(f"Confidence: {sentiment.get('overall_confidence', 0)}")
            formatted.append(f"Articles Analyzed: {sentiment.get('total_articles', 0)}")
        
        if "market_data" in context:
            market = context["market_data"]
            formatted.append(f"\nCurrent Price: ${market.get('current_price', 0)}")
            formatted.append(f"RSI: {market.get('rsi', 'N/A')} - {market.get('rsi_signal', 'N/A')}")
            formatted.append(f"7-Day Trend: {market.get('trend_7d', 'N/A')}")
        
        return "\n".join(formatted)
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
        print("✅ Conversation memory cleared")
    
    def get_relevant_documents(self, query: str, k: int = 5) -> List[Dict]:
        """
        Get relevant documents without generating an answer
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        try:
            retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
            docs = retriever.get_relevant_documents(query)
            
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in docs
            ]
        except Exception as e:
            print(f"❌ Error retrieving documents: {str(e)}")
            return []
    
    def get_database_stats(self) -> Dict:
        """Get statistics about the vector database"""
        try:
            collection = self.vector_store._collection
            count = collection.count()
            
            return {
                "total_documents": count,
                "persist_directory": self.persist_directory,
                "provider": self.llm_provider.upper()
            }
        except:
            return {
                "total_documents": 0,
                "persist_directory": self.persist_directory,
                "provider": self.llm_provider.upper()
            }


# Test function
if __name__ == "__main__":
    print("🧪 Testing RAG Engine with Multiple Providers\n")
    
    test_docs = [
        "Apple Inc. reported strong earnings for Q4 2024, beating analyst expectations with revenue of $120 billion.",
        "Tesla stock surged 15% after announcing new battery technology that could revolutionize electric vehicles.",
        "Market analysts are cautious about tech stocks amid rising interest rates and economic uncertainty."
    ]
    
    # --- Test with Gemini ---
    print("\n" + "="*60)
    print("--- Testing with GEMINI ---")
    print("="*60)
    try:
        gemini_engine = FinancialRAGEngine(llm_provider="gemini")
        
        print("\n📝 Adding test documents (Gemini)...")
        gemini_engine.add_documents(test_docs)
        
        print("\n❓ Testing query (Gemini)...")
        result_gemini = gemini_engine.query("What are the recent developments with Apple and Tesla?")
        
        print(f"\n✅ Gemini Answer: {result_gemini['answer'][:300]}...")
        print(f"   Sources found: {len(result_gemini['sources'])}")
        
        stats_gemini = gemini_engine.get_database_stats()
        print(f"\n📊 Database Stats: {stats_gemini}")
        
    except Exception as e:
        print(f"❌ Gemini Test failed: {str(e)}")

    # --- Test with OpenAI ---
    print("\n" + "="*60)
    print("--- Testing with OPENAI ---")
    print("="*60)
    try:
        openai_engine = FinancialRAGEngine(llm_provider="openai")
        
        print("\n📝 Adding test documents (OpenAI)...")
        openai_engine.add_documents(test_docs)
        
        print("\n❓ Testing query (OpenAI)...")
        result_openai = openai_engine.query("What are the recent developments with Apple and Tesla?")
        
        print(f"\n✅ OpenAI Answer: {result_openai['answer'][:300]}...")
        print(f"   Sources found: {len(result_openai['sources'])}")
        
        stats_openai = openai_engine.get_database_stats()
        print(f"\n📊 Database Stats: {stats_openai}")
        
    except Exception as e:
        print(f"❌ OpenAI Test failed: {str(e)}")
    
    print("\n" + "="*60)
    print("✅ Testing Complete!")
    print("="*60)