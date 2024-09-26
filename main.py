import re
import os
warnings.filterwarnings("ignore")
from openai import AzureOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field
from datetime import datetime
import pandas as pd
import docx2txt
import json
from tqdm import tqdm
from pathlib import Path as p
import urllib
import warnings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.documents import Document
import pandas as pd
import asyncio
import uuid
import tiktoken
from gensim.parsing.preprocessing import remove_stopwords
import operator
from typing import Annotated, List, Literal, TypedDict

from langchain.chains.combine_documents.reduce import (
    acollapse_docs,
    split_list_of_docs,
)
from langchain_core.documents import Document
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
import requests
import gradio as gr
import asyncio
warnings.filterwarnings("ignore")
from dotenv import load_dotenv

encoding = tiktoken.get_encoding("cl100k_base")
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv('AZURE_OPENAI_API_KEY')
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv('AZURE_OPENAI_ENDPOINT')
os.environ["AZURE_OPENAI_API_VERSION"] = os.getenv('AZURE_OPENAI_API_VERSION')
os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = os.getenv('AZURE_OPENAI_CHAT_DEPLOYMENT_NAME')

parameters_prompt = '''You are a sales assistant tasked with summarizing email or phone conversations between a university salesperson and a prospective student. Based on the provided email summary, your goal is to construct a comprehensive student profile, develop a targeted campaign, craft a personalized sales pitch, and assess the student’s interest level in the university's program.

Student Persona
Name: (Student’s name if available, otherwise "NA")
Age: (Age if available otherwise mark as "NA")
Location: (Student's location, or mark as "NA")
Interest: (Summarize the student’s interest in the program or field of study. If not stated, mark as "NA")
Behavior: (Describe any behavioral traits or actions from the emails that indicate the student's approach to education, such as responsiveness, motivation, or any other relevant insights)
Dislikes: (Mention any dislikes expressed in the conversation about education, programs, or formats. Mark as "NA" if unavailable)
Likes: (Include any expressed likes, such as flexibility, subject matter, or specific university features. Mark as "NA" if not mentioned)
Modes of Communication: (Identify all modes of communication between Salesperson and Student)
Preferred Communication: (Identify the student's preferred communication method—email, phone, etc. Mark as "NA" if not specified)

Potential Campaign
Theme: (Craft a campaign theme based on the student’s interests and persona. Example: "Empowering Working Professionals with Flexible Learning")
Content Focus: (Outline the content to feature in this campaign, e.g., "Showcase alumni success stories, emphasize career advancement opportunities, highlight flexible schedules")
Platforms: (List suitable platforms for the campaign, such as "Email marketing, webinars, blog posts, or social media ads")

Sales Strategy
Personalized Sales Pitch: (Write a customized message addressing the student's needs, aspirations, and any concerns.

Sentiment
Student Sentiment: (Assess the overall sentiment from the conversation, such as "Excited", "Curious", "Optimistic", "Anxious", "Confused", "Skeptical", "Indecisive", "Disappointed", "Confident", "Motivated", "Overwhelmed", "Frustrated", "Reluctant", "Satisfied" into a single sentiment.)

Lead Score
Lead Interest Level: (Rate the student’s potential interest in the program: Low, Medium, or High, based on the tone and content of the conversation)'''


llm = AzureChatOpenAI(
api_key=os.environ["AZURE_OPENAI_API_KEY"],
api_version=os.environ["AZURE_OPENAI_API_VERSION"],
azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
model='gpt-4o'
)

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

token_max = 1000


def length_function(documents: List[Document]) -> int:
    """Get number of tokens for input contents."""
    return sum(llm.get_num_tokens(doc.page_content) for doc in documents)


class OverallState(TypedDict):
    contents: List[str]
    summaries: Annotated[list, operator.add]
    collapsed_summaries: List[Document]
    final_summary: str

class SummaryState(TypedDict):
    content: str


async def generate_summary(state: SummaryState):
    response = await map_chain.ainvoke(state["content"])
    return {"summaries": [response]}


def map_summaries(state: OverallState):
    return [
        Send("generate_summary", {"content": content}) for content in state["contents"]
    ]


def collect_summaries(state: OverallState):
    return {
        "collapsed_summaries": [Document(summary) for summary in state["summaries"]]
    }


async def collapse_summaries(state: OverallState):
    doc_lists = split_list_of_docs(
        state["collapsed_summaries"], length_function, token_max
    )
    results = []
    for doc_list in doc_lists:
        results.append(await acollapse_docs(doc_list, reduce_chain.ainvoke))

    return {"collapsed_summaries": results}


def should_collapse(
    state: OverallState,
) -> Literal["collapse_summaries", "generate_final_summary"]:
    num_tokens = length_function(state["collapsed_summaries"])
    if num_tokens > token_max:
        return "collapse_summaries"
    else:
        return "generate_final_summary"


async def generate_final_summary(state: OverallState):
    response = await reduce_chain.ainvoke(state["collapsed_summaries"])
    return {"final_summary": response}


graph = StateGraph(OverallState)
graph.add_node("generate_summary", generate_summary)  # same as before
graph.add_node("collect_summaries", collect_summaries)
graph.add_node("collapse_summaries", collapse_summaries)
graph.add_node("generate_final_summary", generate_final_summary)

graph.add_conditional_edges(START, map_summaries, ["generate_summary"])
graph.add_edge("generate_summary", "collect_summaries")
graph.add_conditional_edges("collect_summaries", should_collapse)
graph.add_conditional_edges("collapse_summaries", should_collapse)
graph.add_edge("generate_final_summary", END)

app = graph.compile()



map_prompt = ChatPromptTemplate.from_messages(
    [("system", "Write a concise one-liner summary of the following conversation, including any student information mentioned, the date of the conversation, the communication channel (phone, email, WhatsApp), the student's level of interest (Low, Medium, or High), and their emotional state (Excited, Curious, Optimistic, Anxious, Confused, Skeptical, Indecisive, Disappointed, Confident, Motivated, Overwhelmed, Frustrated, Neutral, Reluctant, or Satisfied):\n\n{context}")])

map_chain = map_prompt | llm | StrOutputParser()

reduce_template = """
The following are summaries of conversations in no particular order:
{docs}
Please organize these summaries in ascending order by date if available. If dates are not provided, maintain their original order.

"""

reduce_prompt = ChatPromptTemplate([("human", reduce_template)])

reduce_chain = reduce_prompt | llm | StrOutputParser()

API_KEY = load_dotenv('FINE_TUNED_API')
headers = {
    "Content-Type": "application/json",
    "api-key": API_KEY,
}

ENDPOINT = load_dotenv('ENDPOINT')
async def process_data(data):
    docs = [Document(page_content=(data), metadata={"source": "local"})]

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )
    
    split_docs = text_splitter.split_documents(docs)
    print(f"Generated {len(split_docs)} documents.")
    
    async for step in app.astream(
        {"contents": [doc.page_content for doc in split_docs]},
        {"recursion_limit": 10},
    ):
        print(list(step.keys()))
    
    final_summary = step['generate_final_summary']['final_summary']
    
    payload = {
      "messages": [
        {
          "role": "system",
          "content": [
            {
              "type": "text",
              "text": parameters_prompt
            }
          ]
        },
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": final_summary
            }
          ]
        }
      ],
      "temperature": 0.1,
      "top_p": 0.95,
      "max_tokens": 800
    }
    

    try:
        response = requests.post(ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
    except requests.RequestException as e:
        raise SystemExit(f"Failed to make the request. Error: {e}")

    return response.json()


async def main(text):
    num_tokens = num_tokens_from_string(text, "cl100k_base")
    if num_tokens > 8000:
        data = remove_stopwords(text)

    else:
        data = text
    result = await process_data(data)
    return result['choices'][0]['message']['content']


async def process_input_async(input_text):
    return await main(input_text)

iface = gr.Interface(fn=process_input_async, inputs="text", outputs="text", title="Text Processing App")

iface.launch(share=True)