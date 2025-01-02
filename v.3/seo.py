from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma 
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

# Setup embeddings and load documents
local_embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Initialize URLs and load documents
urls = [
    "https://developers.google.com/search/docs/appearance/ai-overviews",
    "https://developers.google.com/search/docs/appearance/ranking-systems-guide",
    "https://developers.google.com/search/docs/appearance/structured-data/sd-policies",
    "https://developers.google.com/search/docs/fundamentals/creating-helpful-content"
]

# Load and process documents
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Split documents
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)

# Create vector store
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=local_embeddings,
)
retriever = vectorstore.as_retriever()

# Create retriever tool
retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and return information about SEO.",
)

# Add DuckDuckGo Search Tool
duckduckgo_tool = DuckDuckGoSearchRun(name="web_search", description="Search the web for information.")

# Define tools
tools = [retriever_tool, duckduckgo_tool]

# Define state management
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Create the memory and model
memory = MemorySaver()
model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)

# Create the agent executor with the graph
from langgraph.prebuilt import create_react_agent
agent_executor = create_react_agent(model, tools, checkpointer=memory)

def process_chat():
    config = {"configurable": {"thread_id": "abc123"}}
    print("Welcome to the SEO Assistant!")
    print("Ask questions about SEO or any topic.")
    print("Type 'exit' to quit.\n")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
            
        if not user_input:
            print("Please enter a valid question.")
            continue
            
        print("\nAssistant: ", end='', flush=True)
        
        # Stream the response
        try:
            for chunk in agent_executor.stream(
                {"messages": [HumanMessage(content=user_input)]}, 
                config
            ):
                # Handle agent messages
                if "agent" in chunk:
                    messages = chunk["agent"].get("messages", [])
                    for message in messages:
                        if isinstance(message, AIMessage):
                            print(message.content, end='', flush=True)
                
                # Handle tool (retrieval and web search) messages
                elif "tools" in chunk:
                    tool_name = chunk.get("tool_name", "Unknown Tool")
                    print(f"\n[Tool Invoked: {tool_name}]")

                    messages = chunk["tools"].get("messages", [])
                    for message in messages:
                        if hasattr(message, 'content'):
                            # Optional: You could choose to print tool outputs for debugging
                            continue
                            
            print("\n")  # Add newline after response
            
        except Exception as e:
            print(f"\nError: {str(e)}\n")

if __name__ == "__main__":
    process_chat()
