import logging

import chainlit as cl
from chainlit.input_widget import Select
from langchain.schema.runnable.config import RunnableConfig

from demoday.state_machine_agent import app, config
from demoday.unstructured_data_chain import unstructured_retrieval_chain

logger = logging.getLogger(__name__)


@cl.set_chat_profiles
async def persona_profile():
    return [
        cl.ChatProfile(
            name="Chat with FPL Data",
            markdown_description="Get the latest stats",
            icon="https://picsum.photos/200",
        ),
        cl.ChatProfile(
            name="FPL Scout Expert Q&A",
            markdown_description="Ask the FPL Scout",
            icon="https://picsum.photos/250",
        ),
    ]


@cl.on_chat_start
async def on_chat_start():
    user = cl.user_session.get("user")
    chat_profile = cl.user_session.get("chat_profile")
    logger.info(f"User: {user}, Chat Profile: {chat_profile}")

    settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="OpenAI - Model",
                values=["gpt-4-turbo", "gpt-3.5-turbo"],
                initial_index=0,
            ),
        ]
    ).send()


async def run_scout_qa(query):

    msg = cl.Message(content="")
    new_line = "\n"

    async for chunk in unstructured_retrieval_chain.astream(
        {"input": query.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        answer = chunk.get("answer")
        if answer:
            await msg.stream_token(answer)
        context = chunk.get("context")
        if context:
            sources = [x.metadata["source"] for x in context]
            sources = f"{new_line}{new_line} Sources:{new_line}" + f"{new_line}".join(
                list(set(sources))
            )

    await cl.Message(content=sources).send()

    return msg.content


@cl.on_message
async def main(query: cl.Message):
    chat_profile = cl.user_session.get("chat_profile")
    logger.info(f"User Input Question: {query.content}")
    if chat_profile == "Chat with FPL Data":
        response = await app.ainvoke(
            {"input": query.content, "chat_history": []}, config
        )
        response = response["agent_outcome"].return_values["output"]
        if response:
            await cl.Message(content=response).send()
    elif chat_profile == "FPL Scout Expert Q&A":
        response = await run_scout_qa(query)
