
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from tavily import TavilyClient
load_dotenv()

@tool
def search_via_internet(query:str):
    """
    query to search using internet """
    client=TavilyClient()
    print(f'searching for query: {query}')
    return client.search(query=query)
    # return "weather is summy and has 36 degrees Celsius"

def my_first_bot_to_search_via_internet():
   llm= ChatGoogleGenerativeAI(model ="gemini-3.6-flash")
   tools =[search_via_internet]
   agent=create_agent(llm,tools ,system_prompt="You are a helpful assistant with internet search capabilities.")
   message=HumanMessage(content = " what is the weather in tokyo")
   result=  agent.invoke({"messages":[message]})
   # result = agent.invoke({"messages": [message]})

   print(f'result: {result['messages'][-1].content}')


if __name__ == '__main__':
    my_first_bot_to_search_via_internet()
