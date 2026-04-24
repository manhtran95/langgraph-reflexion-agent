import datetime

from dotenv import load_dotenv

load_dotenv()

from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser
)

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
# from langchain_ollama import ChatOllama

from schemas import AnswerQuestion, ReviseAnswer

llm = ChatOpenAI(model="o4-mini")
# llm = ChatOllama(model="llama3.1:8b")
# llm = ChatOllama(model="gpt-oss:20b-cloud")
parser = JsonOutputToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher.
            Current time: {time}

            *Your job*
            - Update the AnswerQuestion fields: *answer*, *reflection*, *search_queries*.

            1. *answer* field: Provide a detailed ~250 word answer.
            2. *reflection* field: Reflect and critique your answer.
            3. *search_queries* field: Recommend 2 or 3 search queries to research information and improve your answer.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)

first_responder = actor_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)

# Revisor

revisor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher.
            Current time: {time}

            *Your job*
            - Revise your previous answer using the new information.
            - Update following fields of the ReviseAnswer schema:

            1. *answer* field
            - You should use the previous critique to update information to your answer.
            - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
            2. *references* field
            - Include numerical citations if you know something from urls from previous tool result. Do NOT make up urls. Create a mapping from a number (starting from 1) to some url.
            - Update *references* field of the ReviseAnswer schema with the list of urls.
            Each reference item should be a string in the form of: "[<number>] <url>".            
            3. *reflection* field: Reflect and critique your answer.
            4. *search_queries* field: Recommend 2 or 3 search queries to research information and improve your answer.""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)

revisor = revisor_prompt_template | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")


if __name__ == "__main__":
    human_message = HumanMessage(
        content="Write about LLM domain,"
        " list startups that do that and raised capital."
    )
    chain = (
        actor_prompt_template
        | llm.with_structured_output(AnswerQuestion)
    )

    res = chain.invoke(input={"messages": [human_message]})
    print(res)