from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage,HumanMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAI
# from langchain_
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os
import requests
load_dotenv()
class GrokClient:
    def __init__(self, api_key: str, model: str = "grok-1"):
        self.api_key = api_key
        self.model = model
        # self.url = "https://api.grok.x.ai/v1/generate"  # verify with xAI docs
        self.url = "https://api.openai.com/v1/chat/completions"
    def invoke(self, messages):
        # build prompt from message list (works with BaseMessage/HumanMessage)
        parts = []
        for m in messages:
            parts.append(m.content if hasattr(m, "content") else str(m))
        prompt = "\n".join(parts)

        payload = {"model": self.model, "input": prompt}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        resp = requests.post(self.url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # best-effort extraction of text; adapt to the actual xAI response shape
        return data.get("output") or data.get("text") or data.get("choices", [{}])[0].get("message", {}).get("content", "") or str(data)

# initialize Grok client using env var XAI_API_KEY
llm = GrokClient(api_key=os.getenv("XAI_API_KEY", ""), model=os.getenv("GROK_MODEL", "grok-1"))
# llm = GoogleGenerativeAI(model="gemini-2.5-flash")
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

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