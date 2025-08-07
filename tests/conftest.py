# tests/conftest.py
import datetime
import typing
import zoneinfo

import agents
import openai
import pytest


@pytest.fixture(scope="module")
def model_name():
    """OpenAI model name for testing."""
    return "gpt-4.1-nano"


@pytest.fixture(scope="module")
def openai_client():
    """OpenAI async client for testing."""
    return openai.AsyncOpenAI()


@pytest.fixture(scope="module")
def chat_model(model_name: str, openai_client: openai.AsyncOpenAI):
    """Agents chat model configured with OpenAI client."""
    return agents.OpenAIResponsesModel(model=model_name, openai_client=openai_client)


@pytest.fixture(scope="module")
def model_settings():
    """Model settings for deterministic testing."""
    return agents.ModelSettings(temperature=0.0)


@pytest.fixture(scope="module")
def agent(agents_tool_get_current_time: agents.FunctionTool):
    """Test agent configured with time function tool."""
    return agents.Agent(name="Test Agent", tools=[agents_tool_get_current_time])


@pytest.fixture(scope="module")
def agents_run_config(
    chat_model: agents.OpenAIResponsesModel, model_settings: agents.ModelSettings
):
    """Agents run configuration with tracing disabled."""
    return agents.RunConfig(
        tracing_disabled=True,
        model=chat_model,
        model_settings=model_settings,
    )


@pytest.fixture(scope="module")
def function_get_current_time():
    """Async function that returns current time in Asia/Taipei timezone."""

    async def get_current_time():
        """Get the current time"""
        dt = datetime.datetime.now(zoneinfo.ZoneInfo("Asia/Taipei"))
        dt = dt.replace(microsecond=0)
        return dt.isoformat()

    return get_current_time


@pytest.fixture(scope="module")
def agents_tool_get_current_time(function_get_current_time: typing.Callable[..., str]):
    """Agents function tool wrapper for get_current_time function."""
    return agents.function_tool(function_get_current_time)


@pytest.fixture(scope="module")
def chat_cmpl_tool_get_current_time():
    """OpenAI chat completion tool parameter for get_current_time function."""
    from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
    from openai.types.shared_params.function_definition import FunctionDefinition

    return ChatCompletionToolParam(
        function=FunctionDefinition(
            name="get_current_time", description="Get the current time", parameters={}
        ),
        type="function",
    )
