from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage,HumanMessage
# from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAI,ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()

# llm = GoogleGenerativeAI(model="gemini-2.5-flash")
llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

# Checkpointer
checkpointer = InMemorySaver()

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

# thread_id='1'

# config = {'configurable':{'thread_id': thread_id}}

# response=chatbot.invoke({'messages':[HumanMessage(content="what is capital of india")]},config=config)
# print(response)
# ChatState['response']=response
# # print(ChatState['response'])