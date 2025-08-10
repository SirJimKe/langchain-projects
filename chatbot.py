"""
Simple multi-turn chatbot using LangChain + LangGraph
"""

import os
from dotenv import load_dotenv
import getpass

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")


#LangChain and LangGraph imports
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    BaseMessage,
    trim_messages,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph, MessagesState
from typing import Sequence
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict


model = init_chat_model("gpt-4o-mini", model_provider="openai", api_key=OPENAI_API_KEY)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
         ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Message trimmer
trimmer = trim_messages(
    max_tokens=1024,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str


workflow = StateGraph(state_schema=State)


def call_model(state: State):
    """
    Workflow node that:
    - trims messages
    - constructs prompt
    - calls the model
    - returns the new AI message to be appended
    """
    trimmed = trimmer.invoke(state["messages"])

    prompt =prompt_template.invoke({"messages": trimmed, "language": state["language"]})

    response = model.invoke(prompt)

    return {"messages": [response]}


workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


def run_single_turn(user_text: str, language: str = "English", thread_id: str | None = None):
    """
    Run the app for one user message and return the AIMessage
    """
    input_messages = [HumanMessage(content=user_text)]
    config = {}
    if thread_id:
        config = {"configurable": {"thread_id": thread_id}}

    output = app.invoke({"messages": input_messages, "language": language}, config)

    return output["messages"][-1]


def stream_turn(user_text: str, language: str = "English", thread_id: str | None = None):
    """
    Stream tokens from the app using stream_mode="messages".
    Yields printed tokens as they arrive.
    """
    input_messages = [HumanMessage(content=user_text)]
    config = {}
    if thread_id:
        config = {"configurable": {"thread_id": thread_id}}

    for chunk, metadata in app.stream({"messages": input_messages, "language": language}, config, stream_mode="messages"):
        if isinstance(chunk, AIMessage):
            print(chunk.content, end="", flush=True)
    print()


# Interactive CLI
def interactive_cli():
    print("LangChain Chatbot (type 'exit' to quit)")
    thread = input("Enter a thread id (or press Enter for ephemeral): ").strip()
    if thread == "":
        thread = None
    language = input("Preferred response language (default: English): ").strip() or "English"
    streaming_choice = input("Stream responses? (y/N): ").strip().lower() == "y"

    while True:
        user_text = input("\nYou: ").strip()
        if user_text.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        try:
            if streaming_choice:
                stream_turn(user_text, language, thread_id=thread)
            else:
                ai_msg = run_single_turn(user_text, language, thread_id=thread)

                if hasattr(ai_msg, "content"):
                    print("\nAI:", ai_msg.content)
                else:
                    print("\nAI:", str(ai_msg))
        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    interactive_cli()
