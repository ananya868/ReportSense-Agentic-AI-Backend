"""
Chat worker for the chatbot agent.
This worker is responsible for handling chat messages and generating responses using LLM 
Uses Retrieval Augmented Generation (RAG) to fetch relevant context from the database and generate responses.
"""

import os, sys, json
from pydantic import BaseModel
from typing import List, Dict, Any, Tuple

from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

from openai import OpenAI

from prompts import rag_prompt


class ChatWithDocs: 
    """Chat Worker for the Chatbot agent."""

    def __init__(self, llm_model: str = "gpt-4o-mini", embed_model: str = "text-embedding-3-small", top_k: int = 2) -> None:
        """
        Initialize the chat worker with the given LLM model and embedding model. 
        """
        # Check if OpenAI API key is set
        if not os.getenv("OPENAI_API_KEY"):
            print("Warning: OPENAI_API_KEY environment variable not set")
        
        # Initialize the clients, models
        try:
            self.llm = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
            self.model = llm_model
            self.top_k = top_k
            self.embedding_model = OpenAIEmbeddings(model = embed_model)
            self.vector_store = InMemoryVectorStore(self.embedding_model)
        except Exception as e:
            print(f"Error initializing component: {e}")
            raise exception(f"Error: {e}")
        
        # Conversation State  
        self.conversation_history = []
        self.current_context = None 
        self.last_query_topic = None
        
        # Check if data exists 
        if os.path.exists("data.txt"):
            print("data.txt found!")
            
    def create_followup_prompt(self, query: str) -> str:
        """
        Create a follow-up prompt for the given query.

        Args: 
            query (str): The user query.
            
        Returns: 
            str: The follow-up prompt. 
        """
        # Get the last 3 messages
        recent_exchanges = self.conversation_history[-3:] if len(self.conversation_history) >= 3 else self.conversation_history
        history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent_exchanges]) # Last 3 messages
        
        # Create the follow-up query
        augmented_prompt = f"""
            The last topic on which conversation was held was: {self.last_query_topic}.
        
            Previous conversation:
            {history_text}
            
            Current question: {query}
            
            Please answer the current question in the context of our conversation.
            Make sure to be clear and expressive (Use Emojis) and provide answer in a conversational tone.
            You might use formatting like bullet points, tables, etc. to make the answer more readable.
        """
        return augmented_prompt

    def classify_query(self, query: str) -> Tuple[bool, bool]: 
        """
        Classify the query into follow-up or new query.

        Args:
            query (str): The user query.

        Returns:
            Tuple[bool, bool]: A tuple containing two boolean values:
                - is_followup: True if the query is a follow-up, False otherwise.
                - requires_new_context: True if the query requires new context, False otherwise. 
        """
        assert isinstance(query, str), "Query must be a string"

        # If no conversation history, it's a new query 
        if not self.conversation_history: 
            return False, True
        
        class QueryClassification(BaseModel): 
            is_followup: bool
            requires_new_context: bool

        try: 
            # Create Query Classification Prompt s
            prompt = f"""
                You are part of AI medical chatbot responsible for determining the intent of user queries. 
                The database from which you answer question contains three things: 
                    - Medical Reports (Text based) : These are the medical reports of the user. (and their summaries)
                    - Medical Images (MRIs, Chest X-rays, etc.) findings: These are the findings from the medical images of the user. 
                    - Medicine data : These are the data about the medicines prescribed to the user. (Side effects, dosage, how to use, etc.)
                    - Medicine links : These are buying links for the medicine

                Keeping the above in mind, you are to classify the user query into one of the following categories:
                1. Follow-up query: A query that is a follow-up to the previous conversation and can be answered with existing context.
                2. New query: A query that is not a follow-up to the previous conversation and requires fetching new data.
                
                Analyze this query in the context of the current conversation.
                
                last topic: {self.last_query_topic}
                
                Latest conversation:
                {self.conversation_history[-2]['content'] if len(self.conversation_history) >= 2 else "No previous message"}
                {self.conversation_history[-1]['content'] if len(self.conversation_history) >= 1 else "No previous message"}
                
                New query: {query}
            """
            classification = self.llm.beta.chat.completions.parse(
                model = "gpt-4o-mini",
                messages = [
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                response_format = QueryClassification
            ) 
            is_followup = json.loads(classification.choices[0].message.content)["is_followup"]
            requires_new_context = json.loads(classification.choices[0].message.content)["requires_new_context"]
            return is_followup, requires_new_context

        except Exception as e:
            print(e)
            return False, True

    def intent(self, query: str) -> str:
        """
        Method to handle the intent of the user query and generate response using 'chat' method.

        Args:
            query (str): The user query.

        Returns:
            str: The generated response. 
        """
        assert isinstance(query, str), "Query must be a string"

        is_followup, requires_new_context = self.classify_query(query)

        try:
            if not is_followup or requires_new_context:
                print(":: Fetching new context 🔍 ::")
                # New context has to be fetched
                new_context = self.run_retriever(query)
                self.current_context = new_context
                self.last_query_topic = query

                prompt = rag_prompt.format(query = query, context = self.current_context, history = self.conversation_history)

                response = self.chat(
                    prompt_temp = prompt
                )
            else: 
                print(":: Follow up querying ⤴️ ::")
                # Use existing context
                prompt = self.create_followup_prompt(query)

                response = self.chat(
                    prompt_temp = prompt
                )
        except Exception as e:
            print(e)
            raise exception(f"Error in intent method: {e}")

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": response})
        self.last_query_topic = query

        return response 

    def build_context(self) -> list: 
        """
        Updates the context by reading the data from the file and converting it into langchain documents.

        Returns:
            list: A list of langchain documents.
        """
        try:
            with open("data.txt", "r") as f:
                context = f.read()
            # Process Context    
            chunks = context.split(">>>>")
            # if any element in this list is empty string, remove it 
            chunks = [chunk for chunk in chunks if chunk != '']
            # if any element in this list is None, remove it
            chunks = [chunk for chunk in chunks if chunk is not None]
            # Convert to langchain documents 
            chunk_docs = [Document(page_content=chunk) for chunk in chunks]
            return chunk_docs
        except FileNotFoundError:
            print("data.txt not found")
            return []
        except Exception as e:
            print(f"Error building context: {e}")
            return []
            
    def run_retriever(self, query: str) -> str:
        """
        Run the retriever to get the relevant context for the given query.

        Args: 
            query (str): The user query.
            
        Returns:
            combined_context (str): The combined context from the retrieved documents. 
        """
        assert isinstance(query, str), "Query must be a string"

        # Get context 
        chunk_docs = self.build_context()
        if not chunk_docs: 
            return "No context yet. Please upload medical reports or prescription to get started."
        store = self.vector_store.from_documents(chunk_docs, self.embedding_model)

        # Build retriever 
        retriever = store.as_retriever(search_kwargs={"k": self.top_k})

        # Get relevant documents
        retrieved_docs = retriever.invoke(query)
        # Combine documents
        combined_docs = [doc.page_content for doc in retrieved_docs]
        combined_context = "\n".join(combined_docs)

        return combined_context

    def chat(
        self, 
        prompt_temp: str
    ):  
        """
        Method to generate response using the chat method.

        Args:
            prompt_temp (str): The prompt template.

        Returns:
            str: The generated response. 
        """
        response = self.llm.chat.completions.create(
            model = self.model,
            messages = [
                {
                    "role": "user", 
                    "content": prompt_temp
                }
            ]
        )
        return response.choices[0].message.content
        
        








     
