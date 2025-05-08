import datetime
import sqlite3
from typing import cast

import chainlit as cl
import pandas as pd
import plotly.graph_objects as go
from datasets import DatasetDict, load_dataset
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.tools import DuckDuckGoSearchRun, tool
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from sqlalchemy import create_engine

dataset = cast(DatasetDict, load_dataset("mstz/titanic"))["train"]
titanic_df = cast(pd.DataFrame, dataset.to_pandas())


def _get_engine_for_titanic_db():
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    titanic_df.to_sql(
        "titanic_passengers", connection, if_exists="replace", index=False
    )
    return create_engine("sqlite://", creator=lambda: connection)


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17")
search_tool = DuckDuckGoSearchRun()

db = SQLDatabase(_get_engine_for_titanic_db())
data_toolkit = SQLDatabaseToolkit(db=db, llm=llm)


show_bar_chart_column = None


@tool
def show_bar_chart(column: str):
    """Show a bar chart over the specified column."""
    global show_bar_chart_column
    show_bar_chart_column = column.lower()


agent = create_react_agent(
    llm,
    tools=[search_tool, *data_toolkit.get_tools(), show_bar_chart],
    prompt=f"""
        You are a helpful exploratory data analysis assistant.
        You know everything about the Titanic dataset.
        You can answer questions about the dataset and run SQL queries on it.
        You can answer questions and search the web for information.
        Today is {datetime.date.today()}.
    """,
    checkpointer=InMemorySaver(),
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
    global show_bar_chart_column
    if show_bar_chart_column is not None:
        to_plot = titanic_df[show_bar_chart_column]
        to_plot = to_plot.value_counts(bins=10)
        fig = go.Figure(data=[go.Bar(x=to_plot.index.astype(str), y=to_plot.values)])
        await cl.Message(
            content="",
            elements=[cl.Plotly(figure=fig, display="inline")],
        ).send()
        show_bar_chart_column = None


def main():
    from chainlit.cli import run_chainlit
    from chainlit.config import config as chainlit_config

    chainlit_config.run.watch = True
    run_chainlit(str(__file__))


if __name__ == "__main__":
    main()
