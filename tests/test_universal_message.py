# tests/test_universal_message.py
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
    model_name: str,
    function_get_current_time: typing.Callable[..., typing.Awaitable[str]],
    chat_cmpl_tool_get_current_time: ChatCompletionToolParam,
):
    messages: typing.List[um.Message] = []

    # 1. LLM call: Ask current time
    messages.append(um.Message(role="user", content="What is the current time?"))
    chat_cmpl = await openai_client.chat.completions.create(
        model=model_name,
        messages=um.messages_to_chat_cmpl_messages(messages),
        tools=[chat_cmpl_tool_get_current_time],
    )

    # 2. Assistant request tool call
    assert len(chat_cmpl.choices) == 1
    choice = chat_cmpl.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.tool_calls is not None
    assert len(choice.message.tool_calls) == 1
    tool_call = choice.message.tool_calls[0]
    assert tool_call.function.name == "get_current_time"

    # 3. Sync with `messages`
    messages.append(um.Message.from_any(choice.message))
    assert messages[-1].role == "assistant"
    assert messages[-1].tool_name == "get_current_time"
    assert messages[-1].call_id is not None

    # 4. Execute tool call and add to messages
    tool_output = await function_get_current_time()
    messages.append(
        um.Message(
            role="tool",
            content=tool_output,
            call_id=tool_call.id,
        )
    )

    # 5. Sync with `messages`
    assert messages[-1].role == "tool"
    assert messages[-1].call_id is not None

    # 6. LLM call: Submit messages from `messages` with function call output
    chat_cmpl = await openai_client.chat.completions.create(
        model=model_name,
        messages=um.messages_to_chat_cmpl_messages(messages),
        tools=[chat_cmpl_tool_get_current_time],
    )

    # 7. Validate response
    assert len(chat_cmpl.choices) == 1
    choice = chat_cmpl.choices[0]
    assert choice.finish_reason == "stop"
    assert choice.message.content is not None
    assert "current time" in choice.message.content.lower()

    # 8. Sync with `messages`
    messages.append(um.Message.from_any(choice.message))
    assert messages[-1].role == "assistant"
    assert "current time" in messages[-1].content.lower()

    assert len(messages) == 4


@pytest.mark.asyncio
async def test_agents_tool_usage(
    agent: agents.Agent,  # agents_tool_get_current_time
    agents_run_config: agents.RunConfig,  # chat_model, model_settings
):
    messages: typing.List[um.Message] = []

    # 1. Agent run with asking current time
    messages.append(um.Message(role="user", content="What is the current time?"))
    result: agents.RunResult = await agents.Runner.run(
        agent,
        input=um.messages_to_responses_input_items(messages),
        run_config=agents_run_config,
    )

    # 2. Agents-sdk run function tool call by it self is done by `run`
    # 3. Validate response
    assert isinstance(result.final_output, str)
    assert "current time" in result.final_output.lower()

    # 4. Sync with `messages`
    messages[:] = um.messages_from_any_items(result.to_input_list())
    assert messages[-1].role == "assistant"
    assert "current time" in messages[-1].content.lower()
    assert len(messages) == 4

    # 5. Say thank you with input from `messages`
    messages.append(um.Message(role="user", content="Thank you!"))
    result: agents.RunResult = await agents.Runner.run(
        agent,
        input=um.messages_to_responses_input_items(messages),
        run_config=agents_run_config,
    )

    # 6. Validate response
    assert isinstance(result.final_output, str)

    # 7. Sync with `messages`
    messages[:] = um.messages_from_any_items(result.to_input_list())
    assert messages[-1].role == "assistant"
    assert len(messages) == 6
