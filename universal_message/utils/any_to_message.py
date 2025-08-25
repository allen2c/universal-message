import json
import typing

import durl
import pydantic
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.responses.response_output_text_param import ResponseOutputTextParam

if typing.TYPE_CHECKING:
    from universal_message import SUPPORTED_MESSAGE_TYPES, Message


def from_any(data: "SUPPORTED_MESSAGE_TYPES") -> "Message":
    """Create message from various input types."""
    from universal_message import Message
    from universal_message.utils.chat_cmpl_content_part_to_str import (
        chat_cmpl_content_part_to_str,
    )
    from universal_message.utils.response_input_content_to_str import (
        response_input_content_to_str,
    )
    from universal_message.utils.return_openai_message import (
        return_chat_cmpl_assistant_message,
        return_chat_cmpl_developer_message,
        return_chat_cmpl_function_message,
        return_chat_cmpl_system_message,
        return_chat_cmpl_tool_message,
        return_chat_cmpl_user_message,
        return_response_code_interpreter_tool_call,
        return_response_computer_call_output,
        return_response_computer_tool_call,
        return_response_easy_input_message,
        return_response_file_search_tool_call,
        return_response_function_call_output,
        return_response_function_tool_call,
        return_response_function_web_search,
        return_response_image_generation_call,
        return_response_input_message,
        return_response_item_reference,
        return_response_local_shell_call,
        return_response_local_shell_call_output,
        return_response_mcp_approval_request,
        return_response_mcp_approval_response,
        return_response_mcp_call,
        return_response_mcp_list_tools,
        return_response_output_message,
        return_response_reasoning_item,
    )

    if isinstance(data, Message):
        return data
    if isinstance(data, str):
        return Message(role="user", content=data)
    if isinstance(data, durl.DataURL):
        return Message(role="user", content=str(data))
    if isinstance(data, ChatCompletionMessage):
        if data.tool_calls:
            _tool_call = data.tool_calls[0]  # Only one tool call is supported
            if _tool_call.type == "function":
                return Message(
                    role="assistant",
                    content=(
                        f"[tool_call:{_tool_call.function.name}]"
                        + f"(#{_tool_call.id})"
                        + f":{_tool_call.function.arguments}"
                    ),
                    call_id=_tool_call.id,
                    tool_name=_tool_call.function.name,
                    arguments=_tool_call.function.arguments,
                )
            else:
                raise ValueError(f"Unsupported tool call type: {_tool_call.type}")
        else:
            return Message(
                role="assistant",
                content=data.content or "",
            )
    if isinstance(data, pydantic.BaseModel):
        return Message.model_validate_json(data.model_dump_json())
    if m := return_response_easy_input_message(data):
        return Message(
            role=m["role"],
            content=response_input_content_to_str(m["content"]),
            metadata={"type": "EasyInputMessageParam"},
        )
    if m := return_response_input_message(data):
        return Message(
            role=m["role"],
            content=response_input_content_to_str(m["content"]),
            metadata={"type": "ResponseInputMessageParam"},
        )
    if m := return_response_output_message(data):
        _content_items = m["content"]
        _content_items = typing.cast(
            typing.List[ResponseOutputTextParam], _content_items
        )
        _content = "\n\n".join([c["text"] for c in _content_items])
        return Message(
            id=m["id"],
            role=m["role"],
            content=_content,
            metadata={"type": "ResponseOutputMessageParam"},
        )
    if m := return_response_file_search_tool_call(data):
        _q = "\n\n".join(m["queries"])
        _r = "\n\n".join([r.get("text") or "" for r in m.get("results") or []]).strip()
        return Message(
            id=m["id"],
            role="assistant",
            content=_q + "\n\n" + _r,
            metadata={"type": "ResponseFileSearchToolCallParam"},
        )
    if m := return_response_computer_tool_call(data):
        return Message(
            id=m["id"],
            role="assistant",
            content=json.dumps(m["action"], ensure_ascii=False, default=str),
            call_id=m["call_id"],
            metadata={"type": "ResponseComputerToolCallParam"},
        )
    if m := return_response_computer_call_output(data):
        return Message(
            role="assistant",
            content=json.dumps(m["output"], ensure_ascii=False, default=str),
            call_id=m["call_id"],
            metadata={"type": "ComputerCallOutput"},
        )
    if m := return_response_function_web_search(data):
        return Message(
            id=m["id"],
            role="assistant",
            content=json.dumps(m["action"], ensure_ascii=False, default=str),
            metadata={"type": "ResponseFunctionWebSearchParam"},
        )
    if m := return_response_function_tool_call(data):
        return Message(
            role="assistant",
            content=json.dumps(
                {"name": m["name"], "arguments": m["arguments"]},
                ensure_ascii=False,
                default=str,
            ),
            call_id=m["call_id"],
            tool_name=m["name"],
            arguments=m["arguments"],
            metadata={"type": "ResponseFunctionToolCallParam"},
        )
    if m := return_response_function_call_output(data):
        return Message(
            role="tool",
            content=m["output"],
            call_id=m["call_id"],
            metadata={"type": "FunctionCallOutput"},
        )
    if m := return_response_reasoning_item(data):
        _content = "\n\n".join([s["text"] for s in m["summary"]])
        return Message(
            id=m["id"],
            role="assistant",
            content=_content,
            metadata={"type": "ResponseReasoningItemParam"},
        )
    if m := return_response_image_generation_call(data):
        return Message(
            id=m["id"],
            role="assistant",
            content=m["result"] or "",
            metadata={"type": "ImageGenerationCall"},
        )
    if m := return_response_code_interpreter_tool_call(data):
        _outputs = "\n\n".join(
            [o.get("logs") or o.get("url") or "" for o in m.get("outputs") or []]
        )
        return Message(
            id=m["id"],
            role="assistant",
            content=f"{m['code']}\n\n{_outputs}",
            metadata={"type": "ResponseCodeInterpreterToolCallParam"},
        )
    if m := return_response_local_shell_call(data):
        return Message(
            id=m["id"],
            role="assistant",
            content=json.dumps(m["action"], ensure_ascii=False, default=str),
            call_id=m["call_id"],
            metadata={"type": "LocalShellCall"},
        )
    if m := return_response_local_shell_call_output(data):
        return Message(
            id=m["id"],
            role="assistant",
            content=m["output"],
            metadata={"type": "LocalShellCallOutput"},
        )
    if m := return_response_mcp_list_tools(data):
        return Message(
            id=m["id"],
            role="assistant",
            content=json.dumps(m["tools"], ensure_ascii=False, default=str),
            metadata={"type": "McpListTools"},
        )
    if m := return_response_mcp_approval_request(data):
        return Message(
            id=m["id"],
            role="assistant",
            content=json.dumps(
                {"name": m["name"], "arguments": m["arguments"]},
                ensure_ascii=False,
                default=str,
            ),
            metadata={"type": "McpApprovalRequest"},
        )
    if m := return_response_mcp_approval_response(data):
        return Message(
            role="assistant",
            content=str(m["approve"]),
            call_id=m["approval_request_id"],
            metadata={"type": "McpApprovalResponse"},
        )
    if m := return_response_mcp_call(data):
        return Message(
            id=m["id"],
            role="assistant",
            content=json.dumps(
                {"name": m["name"], "arguments": m["arguments"]},
                ensure_ascii=False,
                default=str,
            ),
            metadata={"type": "McpCall"},
        )
    if m := return_response_item_reference(data):
        return Message(
            id=m["id"],
            role="assistant",
            content="",
            metadata={"type": "ItemReference"},
        )
    if m := return_chat_cmpl_tool_message(data):
        _content = (
            m["content"]
            if isinstance(m["content"], str)
            else chat_cmpl_content_part_to_str(list(m["content"]))
        )
        return Message(
            role="tool",
            content=chat_cmpl_content_part_to_str(_content),
            call_id=m["tool_call_id"],
            metadata={"type": "ChatCompletionToolMessageParam"},
        )
    if m := return_chat_cmpl_user_message(data):
        _content = (
            m["content"]
            if isinstance(m["content"], str)
            else chat_cmpl_content_part_to_str(list(m["content"]))
        )
        if isinstance(_content, list):
            _content = chat_cmpl_content_part_to_str(_content)
        return Message(
            role="user",
            content=_content,
            metadata={"type": "ChatCompletionUserMessageParam"},
        )
    if m := return_chat_cmpl_system_message(data):
        _content = (
            m["content"]
            if isinstance(m["content"], str)
            else chat_cmpl_content_part_to_str(list(m["content"]))
        )
        return Message(
            role="system",
            content=_content,
            metadata={"type": "ChatCompletionSystemMessageParam"},
        )
    if m := return_chat_cmpl_function_message(data):
        return Message(
            role="function",
            content=m["content"] or "",
            metadata={
                "type": "ChatCompletionFunctionMessageParam",
                "name": m["name"],
            },
        )
    if m := return_chat_cmpl_assistant_message(data):
        if _tool_calls := m.get("tool_calls"):
            _tool_call = list(_tool_calls)[0]  # Only one tool call is supported
            if _tool_call["type"] == "function":
                _tool_call_id = _tool_call["id"]
                _tool_call_name = _tool_call["function"]["name"]
                _tool_call_args = _tool_call["function"]["arguments"]
                _content = (
                    f"[tool_call:{_tool_call_name}](#{_tool_call_id})"
                    + f":{_tool_call_args}"
                )
                _content = _content.strip()
                return Message(
                    role="assistant",
                    content=_content,
                    call_id=_tool_call_id,
                    tool_name=_tool_call_name,
                    arguments=_tool_call_args,
                    metadata={"type": "ChatCompletionAssistantMessageParam"},
                )
            else:
                raise ValueError(f"Unsupported tool call type: {_tool_call['type']}")
        else:
            _content = m.get("content") or ""
            _content = (
                _content
                if isinstance(_content, str)
                else "\n\n".join([c.get("text") or "" for c in _content]).strip()
            )
            return Message(
                role="assistant",
                content=_content,
                metadata={"type": "ChatCompletionAssistantMessageParam"},
            )
    if m := return_chat_cmpl_developer_message(data):
        _content = (
            m["content"]
            if isinstance(m["content"], str)
            else chat_cmpl_content_part_to_str(list(m["content"]))
        )
        return Message(
            role="developer",
            content=_content,
            metadata={"type": "ChatCompletionDeveloperMessageParam"},
        )

    return Message.model_validate(data)
