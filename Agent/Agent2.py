""" 
Chatbot 🧠

Objectives:

Use different message types – HumanMessage and AIMessage
Maintain a full conversation history using both message types
Use  model llama-3.1-8b-instant using LangChain's GroqChat
Create a sophisticated conversation loop

Main Goal: Create a form of memory for our Agent

"""

import os
from typing import TypedDict, List, Union
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
  messages: List[Union[HumanMessage, AIMessage]]

llm= ChatGroq(model="llama-3.1-8b-instant")

def process(state: AgentState) -> AgentState:
  """This node will solve the request you input"""
  response= llm.invoke(state["messages"])

  state["messages"].append(AIMessage(content=response.content))
  print(f"AI: {response.content}")
  return state

graph= StateGraph(AgentState)

graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

app= graph.compile()

conversation_history= []

user_input= input("Enter: ")
while user_input.lower()!= "exit":
  conversation_history.append(HumanMessage(content=user_input)) # Add the user input to the conversation history
  result= app.invoke({"messages": conversation_history}) # Invoke the app with the updated conversation history

  print(result["messages"])
  conversation_history= result["messages"] # Update the conversation history with the new messages
  user_input= input("User: ")


