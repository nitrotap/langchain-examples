from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma 
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import OllamaEmbeddings, ChatOllama
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
    "Retreive information on how to write a good blog post and technology article geared towards SEO",
)

# Add Bing Search Tool
def bing_search(query: str):
    """
    Perform a Bing search and return the top search results.
    
    Args:
        query (str): The search query.
    
    Returns:
        List[Dict[str, str]]: A list of search results, where each result contains
                              a title, link, and snippet.
    """
    import requests
    from bs4 import BeautifulSoup

    url = f"https://www.bing.com/search?q={query}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return [{"error": f"Failed to fetch search results. Status code: {response.status_code}"}]

    soup = BeautifulSoup(response.text, "html.parser")
    results = []

    for item in soup.select(".b_algo, .b_imgcap_altitle"):
        title_tag = item.select_one("h2 a")
        snippet_tag = item.select_one(".b_caption p")

        if title_tag and title_tag.get("href"):
            results.append({
                "title": title_tag.text.strip(),
                "link": title_tag["href"],
                "snippet": snippet_tag.text.strip() if snippet_tag else ""
            })

    return results

from langchain_community.tools.arxiv.tool import ArxivQueryRun
arxiv_tool = ArxivQueryRun()
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# Define tools
tools = [retriever_tool, bing_search, arxiv_tool, wikipedia]



# Define state management
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Create the memory and model
memory = MemorySaver()
# model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)
model = ChatOllama(temperature=0.1, model="llama3.2")


# Create the agent executor with the graph
from langgraph.prebuilt import create_react_agent

# agent_executor = create_react_agent(model, tools, checkpointer=memory)

from langchain_core.prompts import PromptTemplate
# Define the custom prompt template
# Import date class from datetime module
from datetime import date

# Returns the current local date
today = date.today()

template = '''
You are an intelligent assistant designed to answer questions and write blog posts by effectively utilizing multiple tools. The current date is {today}.
You have access to the following tools:
1. Retriever Tool - Search and return information from the document corpus pertaining to creating high quality posts for SEO.
2. Bing Web Search - Search the web for general information.
3. Arxiv Tool - Search and query academic papers for research-based knowledge about AI.
4. Wikipedia Tool - Retrieve information from Wikipedia.

Guidelines for using tools:
- Always consider combining multiple tools when a single tool might not provide a comprehensive answer. For example, always use the retriever tool to search for information on writing blog posts in combination with another tool.
- If the first tool's output is incomplete or unclear, proceed to query additional tools to cross-verify or expand the information.
- When answering complex questions, always use a combination of tools to create a well-rounded response.

Your goal is to provide accurate, detailed, and well-supported answers by intelligently combining the outputs from multiple tools when necessary.

'''

# Create the PromptTemplate
# prompt = PromptTemplate.from_template(template)

# Initialize the agent with the custom prompt
agent_executor = create_react_agent(
    model,
    tools,
    checkpointer=memory,
    state_modifier=template
)

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
                    messages = chunk["tools"].get("messages", [])
                    for message in messages:
                        if hasattr(message, 'content'):
                            # Optional: You could choose to print tool outputs for debugging
                            print(message)
                            print(chunk)
                            continue
                            
            print("\n")  # Add newline after response
            
        except Exception as e:
            print(f"\nError: {str(e)}\n")

if __name__ == "__main__":
    process_chat()
