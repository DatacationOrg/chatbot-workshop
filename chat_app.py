import datetime

import chainlit as cl
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

search_tool = DuckDuckGoSearchRun()

agent = create_react_agent(
    ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17"),
    tools=[search_tool],
    prompt=f"""
        You are a helpful assistant.
        You can answer questions and search the web for information.
        Today is {datetime.date.today()}.
    """,
    checkpointer=MemorySaver(),
)


@cl.on_message
async def on_message(message: cl.Message):
    final_answer = cl.Message(content="")
    langgraph_step = None
    async for msg, metadata in agent.astream(
        dict(messages=message.content),
        stream_mode="messages",
        config=RunnableConfig(
            callbacks=[cl.AsyncLangchainCallbackHandler()],
            configurable=dict(thread_id=cl.user_session.get("id")),
        ),
    ):
        if (
            isinstance(msg, AIMessageChunk)
            and isinstance(metadata, dict)
            and metadata["langgraph_node"] == "agent"
            and isinstance(msg.content, str)
        ):
            current_step = metadata["langgraph_step"]
            if current_step != langgraph_step:
                if langgraph_step is not None:
                    await final_answer.stream_token("\n")
                langgraph_step = current_step
            await final_answer.stream_token(msg.content)
    await final_answer.send()


def main():
    from chainlit.cli import run_chainlit
    from chainlit.config import config as chainlit_config

    chainlit_config.run.watch = True
    run_chainlit(str(__file__))


if __name__ == "__main__":
    main()
