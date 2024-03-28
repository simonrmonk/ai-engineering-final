from langchain.tools import BaseTool, StructuredTool, tool
import logging
import operator
from typing import Annotated, List, Literal, Sequence, TypedDict, Union

import chainlit as cl
import pandas as pd
from langchain.agents import create_openai_functions_agent
from langchain.tools import tool
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.vectorstores import Chroma
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor
from pydantic import BaseModel, Field

from . import llm, new_line_character
from .few_shot_agent_prompt import full_sql_prompt
from .human_as_a_tool import chain_get_human_question, chain_get_noun
from .player_stats_data import db, retrievers
from .unstructured_data_chain import unstructured_retrieval_chain


async def ask_human(original_user_inputs, options):

    question = chain_get_human_question.invoke(
        {"original_user_inputs": original_user_inputs, "options": options}
    ).content
    logging.info(f"Question to human: {question}")

    try:
        human_response = await cl.AskUserMessage(content=question, timeout=20).send()
        human_response = human_response["output"]
        nouns = chain_get_noun.invoke(
            {"question_to_user": options, "user_response": human_response}
        )
    except:
        nouns = "NA"

    if nouns:
        return question, nouns


class NounLookupInput(BaseModel):
    column: Literal["full_name"] = Field(
        description="This is the column that the user wants to filter on. For example 'full_name'"
    )
    inputs: List[str] = Field(
        description="This is a list of strings that the user passed in to use for the column filter. For example ['Raheem', 'Salah']"
    )


class NounLookupInputWrapper(BaseModel):
    tool_input: List[NounLookupInput]


from typing import List

async def noun_lookup(tool_input) -> List[List[str]]:
    outputs = list()
    for input2 in tool_input:
        column = input2.column
        inputs = input2.inputs
        retriever = retrievers[column]
        options = list()
        for x in inputs:
            o = retriever.get_relevant_documents(x)
            o = [option.page_content for option in o]
            options.append(o)
        question, noun = await ask_human(inputs, options)
        temp = {column: noun.content.split('\n')}
        outputs.append(temp)
    return outputs

description = """
This tool filters columns and requests clarification on the specific noun to filter on.
It outputs confirmation of selected nouns and their respective columns or provides feedback and a question if clarification is needed.

A response of 'NA' indicates either no selection was made by the user or no semantically similar nouns were found.
In the even of an 'NA', you absolutely must ask a follow-up question of the user to clarify what they want.

Example input: [{'column': 'full_name', 'inputs': ['Raheem', 'Salah']}, {'column': 'book_name', 'inputs': ['Harry', 'Girl with the']}]
Example output: [{'full_name': ['Raheem Sterling', 'Mohamed Salah']}, {'book_name': ['Harry Potter', 'Girl with the Dragon Tattoo']}]

Ensure the exact noun used is provided in the final output for transparency.
"""

noun_lookup_tool = StructuredTool.from_function(
    name="noun_lookup_tool",
    description=description,
    args_schema=NounLookupInputWrapper,
    return_direct=True,
    coroutine=noun_lookup
)  

# @tool(args_schema=NounLookupInputWrapper)
# async def noun_lookup_tool(tool_input) -> List[List[str]]:
#     """
#     This tool must be called when we want to filter on columns.
#     The inputs to this are a list of dictionaries, where each item is for a specific column and nouns to filter on for that column.
#     The input is the original spelling that the user put in for each noun.
#     The tool will ask the human for clarification on which exact noun they want to filter on.
#     The output of this tool is either some sort of confirmation of the nouns and their columns, or feedback on what to change.
#     If the response is 'NA', it means the user didn't want any of the options OR there were no nouns that were scemantically similar enough.

#     Example input: [{'column': 'full_name', 'inputs': ['Raheem', 'Salah']}]
#     Example output: [['Raheem Sterling', 'Mohamed Salah']]

#     Make sure that when you use this tool, you provide the exact noun name you ended up using in your final output for transparency.
#     """
#     for input2 in tool_input:
#         column = input2.column
#         inputs = input2.inputs
#         retriever = retrievers[column]
#         options = list()
#         for x in inputs:
#             o = retriever.get_relevant_documents(x)
#             o = [option.page_content for option in o]
#             options.append(o)
#         # options = [option.page_content for option in options]
#         noun = await ask_human(column, options)
#     return noun


def _get_agent_state(input_schema=None):
    if input_schema is None:

        class AgentState(TypedDict):
            # The input string
            input: str
            # The list of previous messages in the conversation
            chat_history: Sequence[BaseMessage]
            # The outcome of a given call to the agent
            # Needs `None` as a valid type, since this is what this will start as
            agent_outcome: Union[AgentAction, AgentFinish, None]
            # List of actions and corresponding observations
            # Here we annotate this with `operator.add` to indicate that operations to
            # this state should be ADDED to the existing values (not overwrite it)
            intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

    else:

        class AgentState(input_schema):
            # The outcome of a given call to the agent
            # Needs `None` as a valid type, since this is what this will start as
            agent_outcome: Union[AgentAction, AgentFinish, None]
            # List of actions and corresponding observations
            # Here we annotate this with `operator.add` to indicate that operations to
            # this state should be ADDED to the existing values (not overwrite it)
            intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

    return AgentState


def create_agent_executor(agent_runnable, tools, input_schema=None):
    if isinstance(tools, ToolExecutor):
        tool_executor = tools
    else:
        tool_executor = ToolExecutor(tools)

    state = _get_agent_state(input_schema)

    # Define logic that will be used to determine which conditional edge to go down

    def should_continue(data):
        # If the agent outcome is an AgentFinish, then we return `exit` string
        # This will be used when setting up the graph to define the flow
        if isinstance(data["agent_outcome"], AgentFinish):
            return "end"
        # Otherwise, an AgentAction is returned
        # Here we return `continue` string
        # This will be used when setting up the graph to define the flow
        else:
            return "continue"

    def run_agent(data):
        agent_outcome = agent_runnable.invoke(data)
        return {"agent_outcome": agent_outcome}

    async def arun_agent(data):
        agent_outcome = await agent_runnable.ainvoke(data)
        return {"agent_outcome": agent_outcome}

    # Define the function to execute tools
    def execute_tools(data):
        # Get the most recent agent_outcome - this is the key added in the `agent` above
        agent_action = data["agent_outcome"]
        if isinstance(agent_action, list):
            output = tool_executor.batch(agent_action, return_exceptions=True)
            return {
                "intermediate_steps": [
                    (action, str(out)) for action, out in zip(agent_action, output)
                ]
            }
        output = tool_executor.invoke(agent_action)
        return {"intermediate_steps": [(agent_action, str(output))]}

    async def aexecute_tools(data):
        # Get the most recent agent_outcome - this is the key added in the `agent` above
        agent_action = data["agent_outcome"]
        if isinstance(agent_action, list):
            output = list()
            for action in agent_action:
                out = await tool_executor.ainvoke(action)
                output.append(out)
            return {
                "intermediate_steps": [
                    (action, str(out)) for action, out in zip(agent_action, output)
                ]
            }

        output = await tool_executor.ainvoke(agent_action)
        return {"intermediate_steps": [(agent_action, str(output))]}

    # Define a new graph
    workflow = StateGraph(state)

    # Define the two nodes we will cycle between
    # workflow.add_node("first_agent", first_model)
    workflow.add_node("agent", RunnableLambda(run_agent, arun_agent))
    workflow.add_node("action", RunnableLambda(execute_tools, aexecute_tools))
    # workflow.add_node("human", RunnableLambda(ask_human))

    # Set the entrypoint as `agent`
    # This means that this node is the first one called
    workflow.set_entry_point("agent")
    # workflow.set_entry_point("first_agent")

    # We now add a conditional edge
    workflow.add_conditional_edges(
        # First, we define the start node. We use `agent`.
        # This means these are the edges taken after the `agent` node is called.
        "agent",
        # Next, we pass in the function that will determine which node is called next.
        should_continue,
        {
            # If `tools`, then we call the tool node.
            "continue": "action",
            # Otherwise we finish.
            "end": END,
        },
    )

    workflow.add_edge("action", "agent")

    return workflow.compile()


toolkit = SQLDatabaseToolkit(db=db, llm=llm)

tools = toolkit.get_tools()


class UserStrategyQuery(BaseModel):
    input: str = Field(
        description="This is a query from the user about general strategy or strategy regarding the current game week"
    )


@tool
def strategy_advisor_tool(query: str):
    """This is a fantastic tool for recommendations for the current game week as well as general FPL strategic questions."""
    return unstructured_retrieval_chain.invoke({"input": query})


tools = tools + [noun_lookup_tool]


from langchain import hub

prompt = hub.pull("hwchase17/openai-functions-agent")


agent_runnable = create_openai_functions_agent(llm, tools, full_sql_prompt)
app = create_agent_executor(agent_runnable, tools)


config = RunnableConfig(recursion_limit=100)
