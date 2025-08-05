# tests/conftest.py
import datetime
import typing
import zoneinfo

import agents
import openai
import pytest


@pytest.fixture(scope="module")
def model_name():
    return "gpt-4.1-nano"


@pytest.fixture(scope="module")
def openai_client():
    return openai.AsyncOpenAI()


@pytest.fixture(scope="module")
def chat_model(model_name: str, openai_client: openai.AsyncOpenAI):
    return agents.OpenAIResponsesModel(model=model_name, openai_client=openai_client)


@pytest.fixture(scope="module")
def model_settings():
    return agents.ModelSettings(temperature=0.0)


@pytest.fixture(scope="module")
def agent(agents_tool_get_current_time: agents.FunctionTool):
    return agents.Agent(name="Test Agent", tools=[agents_tool_get_current_time])


@pytest.fixture(scope="module")
def agents_run_config(
    chat_model: agents.OpenAIResponsesModel, model_settings: agents.ModelSettings
):
    return agents.RunConfig(
        tracing_disabled=True,
        model=chat_model,
        model_settings=model_settings,
    )


@pytest.fixture(scope="module")
def function_get_current_time():
    async def get_current_time():
        """Get the current time"""
        dt = datetime.datetime.now(zoneinfo.ZoneInfo("Asia/Taipei"))
        dt = dt.replace(microsecond=0)
        return dt.isoformat()

    return get_current_time


@pytest.fixture(scope="module")
def agents_tool_get_current_time(function_get_current_time: typing.Callable[..., str]):
    return agents.function_tool(function_get_current_time)


@pytest.fixture(scope="module")
def chat_cmpl_tool_get_current_time():
    from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
    from openai.types.shared_params.function_definition import FunctionDefinition

    return ChatCompletionToolParam(
        function=FunctionDefinition(
            name="get_current_time", description="Get the current time", parameters={}
        ),
        type="function",
    )
