import chainlit as cl
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17")


@cl.on_message
async def on_message(message: cl.Message):
    msg = cl.Message(content="")
    async for chunk in llm.astream(
        message.content,
        config=RunnableConfig(callbacks=[cl.AsyncLangchainCallbackHandler()]),
    ):
        chunk_content = chunk.content
        assert isinstance(chunk_content, str)
        await msg.stream_token(chunk_content)
    await msg.send()


def main():
    from chainlit.cli import run_chainlit
    from chainlit.config import config as chainlit_config

    chainlit_config.run.watch = True
    run_chainlit(str(__file__))


if __name__ == "__main__":
    main()
