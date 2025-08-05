"""Test compatibility of universal_message.Message objects with openai-agents chat and chat completion usages.

All messages are stored in universal_message.Message objects.
Convert to `typing.List[ResponseInputItemParam]` by `messages_to_responses_input_items` function when testing with openai-agents chat.
Convert to `typing.List[ChatCompletionMessageParam]` by `messages_to_chat_cmpl_messages` function when testing with openai-agents chat.
Convert back to `universal_message.Message` objects by `messages_from_any_items` function after LLM chat completion.
All tests should call function `get_current_time` once.
All tests should call llm twice including submitted function call output.
"""  # noqa: E501

import typing

import agents
import openai
import pytest
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

import universal_message as um


@pytest.mark.asyncio
async def test_chat_cmpl_tool_usage(
    openai_client: openai.AsyncOpenAI,
    function_get_current_time: typing.Callable[..., str],
    chat_cmpl_tool_get_current_time: ChatCompletionToolParam,
):
    messages: typing.List[um.Message] = []

    # 1. LLM call: Ask current time

    # 2. Assistant request tool call

    # 3. Sync with `messages`

    # 4. Execute tool call and add to messages

    # 5. Sync with `messages`

    # 6. LLM call: Submit messages from `messages` with function call output

    # 7. Validate response

    # 8. Sync with `messages`

    assert messages


@pytest.mark.asyncio
async def test_agents_tool_usage(
    chat_model: agents.OpenAIResponsesModel,
    model_settings: agents.ModelSettings,
    agent: agents.Agent,
    agents_run_config: agents.RunConfig,
    agents_tool_get_current_time: agents.FunctionTool,
):
    messages: typing.List[um.Message] = []

    # 1. Agent run with asking current time

    # 2. Agents-sdk run function tool call by it self

    # 3. Validate response

    # 4. Sync with `messages`

    # 5. Say thank you with input from `messages`

    # 6. Validate response

    # 7. Sync with `messages`

    assert messages
