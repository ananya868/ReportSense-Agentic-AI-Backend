import os
import argparse 
from chat_worker import ChatWithDocs
from dotenv import load_dotenv

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="CLI chatbot with document retrieval")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model to use")
    parser.add_argument("--embedding", default="text-embedding-3-small", help="Embedding model to use")
    parser.add_argument("--top-k", type=int, default=2, help="Number of documents to retrieve")
    args = parser.parse_args()

    # Initialize the chatbot
    print(f"Initializing chatbot with model: {args.model}")
    chatbot = ChatWithDocs(
        llm_model=args.model,
        embed_model=args.embedding,
        top_k=args.top_k
    )  

    print("\n=== 📑 ChatWithDocs CLI 📑 ===")
    print("Type 'exit', 'quit', or press Ctrl+C to end the conversation.\n")
    
    try:
        while True:
            user_input = input("\n👦 You => ")
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
                
            # Get response
            print("\n🧠🤔 Thinking . . . ")
            response = chatbot.intent(user_input)
            
            print(f"\n🤖 Assistant => {response}")
            
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        
if __name__ == "__main__":
    main()