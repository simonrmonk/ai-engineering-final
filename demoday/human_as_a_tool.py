from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

from . import llm

new_line_character = "\n"
examples_noun_id = [
    {
        "question_to_user": f"There are multiple player names that match that query, please choose one:{new_line_character}{new_line_character.join(['Raheem Sterling', 'Sterling Silver', 'Harry Kane', 'Mohamed Salah'])}",
        "user_response": "Harry",
        "output": "Harry Kane",
    },
    {
        "question_to_user": f"There are multiple player names that match that query, please choose one:{new_line_character}{new_line_character.join(['Raheem Sterling', 'Sterling Silver', 'Harry Kane', 'Mohamed Salah'])}",
        "user_response": "the last one",
        "output": "Mohamed Salah",
    },
    {
        "question_to_user": f"There are multiple player names that match that query, please choose one:{new_line_character}{new_line_character.join(['Raheem Sterling', 'Sterling Silver', 'Harry Kane', 'Mohamed Salah'])}",
        "user_response": "none of those",
        "output": "NA",
    },
    {
        "question_to_user": 'You made two name references - Simon and James, could you confirm that\n\n1) you were refering to Simon Monk? and\n2) which of "James Roderick", "Jordan Henderson", "Harry Kane" you are refering to?',
        "user_response": "Yes and James",
        "output": "Simon Monk\nJames Roderick",
    },
    {
        "question_to_user": 'You made two name references - Simon and James, could you confirm that\n\n1) you were refering to Simon Monk? and\n2) which of "James Roderick", "Jordan Henderson", "Harry Kane" you are refering to?',
        "user_response": "The first one in both cases",
        "output": "Simon Monk\nJames Roderick",
    },
    {
        "question_to_user": 'You made two name references - Simon and James, could you confirm that\n\n1) you were refering to Simon Monk? and\n2) which of "James Roderick", "Jordan Henderson", "Harry Kane" you are refering to?',
        "user_response": "1. Yes 2. Harry",
        "output": "Simon Monk\nHarry Kane",
    },
]

example_prompt = PromptTemplate(
    input_variables=["question_to_user", "user_response", "output"],
    template="Question to User: {question_to_user}\n'User Response': {user_response}\n{output}",
)

system = """You will be provided a list of options a human has to choose from.
Your an expect at identifying, given a user input, which of the options they are refering to.
You must only respond with the exact spelling of the option the user is refering to.
The only exception is when you're not sure which option the user is refering to or the user indicates
that the option they want is something else, in which case you should respond with 'NA'.
"""

few_shot_prompt = FewShotPromptTemplate(
    examples=examples_noun_id,
    example_prompt=example_prompt,
    suffix=system
    + "Question to User: {question_to_user}\n'User Response': {user_response}",
    input_variables=["question_to_user", "user_response"],
)


get_noun_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(prompt=few_shot_prompt),
        (
            "human",
            "Question to User: {question_to_user}\n'User Response': {user_response}",
        ),
    ]
)

chain_get_noun = get_noun_prompt | llm

new_line_character = "\n"
examples_noun_id = [
    {
        "original_user_inputs": ["Simon", "James"],
        "options": [
            ["Simon Monk"],
            ["James Roderick", "Jordan Henderson", "Harry Kane"],
        ],
        "question": 'You made two name references - Simon and James, could you confirm that\n\n1) you were refering to Simon Monk? and\n2) which of "James Roderick", "Jordan Henderson", "Harry Kane" you are refering to?',
    },
]

example_prompt = PromptTemplate(
    input_variables=["original_user_inputs", "options", "question"],
    template="Original User Inputs: {original_user_inputs}\n'Options': {options}\n{question}",
)

system = """You're going to be provided with the original user inputs and a list of options that correspond to each of the user inputs.
Your job is to phrase a clean and consise question for the user to clarify which of the options they are refering to, for each of their original inputs (Make sure to use new lines to achieve this).
"""

few_shot_prompt2 = FewShotPromptTemplate(
    examples=examples_noun_id,
    example_prompt=example_prompt,
    suffix=system
    + "Original User Inputs: {original_user_inputs}\n'User Response': {options}",
    input_variables=["original_user_inputs", "options"],
)


get_question_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(prompt=few_shot_prompt2),
        (
            "human",
            "Original User Inputs: {original_user_inputs}\n'User Response': {options}",
        ),
    ]
)

chain_get_human_question = get_question_prompt | llm
