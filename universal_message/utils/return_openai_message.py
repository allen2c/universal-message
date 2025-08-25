# universal_message/utils/return_openai_message.py
import typing

from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)
from openai.types.chat.chat_completion_developer_message_param import (
    ChatCompletionDeveloperMessageParam,
)
from openai.types.chat.chat_completion_function_message_param import (
    ChatCompletionFunctionMessageParam,
)
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from openai.types.chat.chat_completion_tool_message_param import (
    ChatCompletionToolMessageParam,
)
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)
from openai.types.responses.easy_input_message_param import EasyInputMessageParam
from openai.types.responses.response_code_interpreter_tool_call_param import (
    ResponseCodeInterpreterToolCallParam,
)
from openai.types.responses.response_computer_tool_call_output_screenshot_param import (
    ResponseComputerToolCallOutputScreenshotParam,
)
from openai.types.responses.response_computer_tool_call_param import (
    ResponseComputerToolCallParam,
)
from openai.types.responses.response_file_search_tool_call_param import (
    ResponseFileSearchToolCallParam,
)
from openai.types.responses.response_function_tool_call_param import (
    ResponseFunctionToolCallParam,
)
from openai.types.responses.response_function_web_search_param import (
    ResponseFunctionWebSearchParam,
)
from openai.types.responses.response_input_item_param import (
    ComputerCallOutput,
    FunctionCallOutput,
    ImageGenerationCall,
    ItemReference,
    LocalShellCall,
    LocalShellCallOutput,
    McpApprovalRequest,
    McpApprovalResponse,
    McpCall,
    McpListTools,
)
from openai.types.responses.response_input_item_param import (
    Message as ResponseInputMessageParam,
)
from openai.types.responses.response_output_message_param import (
    ResponseOutputMessageParam,
)
from openai.types.responses.response_reasoning_item_param import (
    ResponseReasoningItemParam,
)

if typing.TYPE_CHECKING:
    from universal_message import OPENAI_MESSAGE_PARAM_TYPES


__all__ = [
    "return_response_easy_input_message",
    "return_response_input_message",
    "return_response_output_message",
    "return_chat_cmpl_user_message",
    "return_chat_cmpl_system_message",
    "return_chat_cmpl_function_message",
    "return_chat_cmpl_assistant_message",
    "return_chat_cmpl_developer_message",
]


def return_response_easy_input_message(
    message: "OPENAI_MESSAGE_PARAM_TYPES",
) -> EasyInputMessageParam | None:
    """Returns an EasyInputMessageParam if the message is a valid easy input message."""
    from universal_message.utils.is_response_input_message_content_list_param import (
        is_response_input_message_content_list_param,
    )

    # Check required fields
    if "role" not in message or "content" not in message:
        return None
    # Check type: message
    if message.get("type") != "message":
        return None
    # Check roles
    if message["role"] not in ("user", "assistant", "system", "developer"):
        return None
    if message.get("status"):  # go `ResponseInputMessageParam`
        return None
    # Check content: list of input items
    if isinstance(message["content"], str):
        return message  # type: ignore
    elif isinstance(message["content"], list):
        if is_response_input_message_content_list_param(message["content"]):  # type: ignore  # noqa: E501
            return message  # type: ignore
        else:
            return None
    else:
        return None


def return_response_input_message(
    message: "OPENAI_MESSAGE_PARAM_TYPES",
) -> ResponseInputMessageParam | None:
    """Returns a ResponseInputMessageParam if the message is a valid input message."""
    from universal_message.utils.is_response_input_message_content_list_param import (
        is_response_input_message_content_list_param,
    )

    # Check required fields
    if "role" not in message or "content" not in message:
        return None
    # Check type: message
    if message.get("type") != "message":
        return None
    # Check roles
    if message["role"] not in ("user", "system", "developer"):
        return None
    # Check content: list of input items
    if isinstance(message["content"], list):
        if is_response_input_message_content_list_param(message["content"]):  # type: ignore  # noqa: E501
            return message  # type: ignore
        else:
            return None
    else:
        return None


def return_response_output_message(
    message: "OPENAI_MESSAGE_PARAM_TYPES",
) -> ResponseOutputMessageParam | None:
    """Returns a ResponseOutputMessageParam if the message is a valid output message."""
    from universal_message.utils.is_response_output_text_param import (
        is_response_output_text_param,
    )

    if (
        "id" not in message
        or "content" not in message
        or "role" not in message
        or "status" not in message
        or "type" not in message
    ):
        return None
    if message["role"] != "assistant":
        return None
    if message["status"] not in ("in_progress", "completed", "incomplete"):
        return None
    if message["type"] != "message":
        return None
    if isinstance(message["content"], list):
        if all(is_response_output_text_param(item) for item in message["content"]):  # type: ignore  # noqa: E501
            return message  # type: ignore
        else:
            return None
    else:
        return None


def return_response_file_search_tool_call(
    message: "OPENAI_MESSAGE_PARAM_TYPES",
) -> ResponseFileSearchToolCallParam | None:
    """Returns ResponseFileSearchToolCallParam if message is valid file search call.
    Validates required fields and status values.
    """
    if (
        "id" not in message
        or "queries" not in message
        or "status" not in message
        or "type" not in message
    ):
        return None
    if message["status"] not in (
        "in_progress",
        "searching",
        "completed",
        "incomplete",
        "failed",
    ):
        return None
    if message["type"] != "file_search_call":
        return None
    return message  # type: ignore


def return_response_computer_tool_call(
    message: "OPENAI_MESSAGE_PARAM_TYPES",
) -> ResponseComputerToolCallParam | None:
    """Returns ResponseComputerToolCallParam if message is valid computer call.
    Validates required fields and status values.
    """
    if (
        "id" not in message
        or "action" not in message
        or "call_id" not in message
        or "pending_safety_checks" not in message
        or "status" not in message
        or "type" not in message
    ):
        return None
    if message["type"] != "computer_call":
        return None
    if message["status"] not in ("in_progress", "completed", "incomplete"):
        return None
    return message  # type: ignore


def return_response_computer_call_output(
    message: "OPENAI_MESSAGE_PARAM_TYPES",
) -> ComputerCallOutput | None:
    """Returns a ComputerCallOutput if the message is a valid computer call output."""
    if "call_id" not in message or "output" not in message or "type" not in message:
        return None
    if message["type"] != "computer_call_output":
        return None
    if "status" in message and message["status"] not in (
        "in_progress",
        "completed",
        "incomplete",
    ):
        return None
    return message  # type: ignore


def return_response_function_web_search(
    message: "OPENAI_MESSAGE_PARAM_TYPES",
) -> ResponseFunctionWebSearchParam | None:
    """Returns ResponseFunctionWebSearchParam if message is valid web search.
    Validates required fields and status values.
    """
    if (
        "id" not in message
        or "action" not in message
        or "status" not in message
        or "type" not in message
    ):
        return None
    if message["type"] != "web_search_call":
        return None
    if message["status"] not in ("in_progress", "searching", "completed", "failed"):
        return None
    return message  # type: ignore


def return_response_function_tool_call(
    message: "OPENAI_MESSAGE_PARAM_TYPES",
) -> ResponseFunctionToolCallParam | None:
    """Returns ResponseFunctionToolCallParam if message is valid function call.
    Validates required fields and status values.
    """
    if (
        "arguments" not in message
        or "call_id" not in message
        or "name" not in message
        or "type" not in message
    ):
        return None
    if message["type"] != "function_call":
        return None
    if "status" in message and message["status"] not in (
        "in_progress",
        "completed",
        "incomplete",
    ):
        return None
    return message  # type: ignore


def return_response_function_call_output(
    message: "OPENAI_MESSAGE_PARAM_TYPES",
) -> FunctionCallOutput | None:
    """Returns FunctionCallOutput if message is valid function call output.
    Validates required fields and status values.
    """
    if "call_id" not in message or "output" not in message or "type" not in message:
        return None
    if message["type"] != "function_call_output":
        return None
    if "status" in message and message["status"] not in (
        "in_progress",
        "completed",
        "incomplete",
    ):
        return None
    return message  # type: ignore


def return_response_reasoning_item(
    message: "OPENAI_MESSAGE_PARAM_TYPES",
) -> ResponseReasoningItemParam | None:
    """Returns ResponseReasoningItemParam if the message is valid reasoning item.
    Validates required fields and status values.
    """
    if "id" not in message or "summary" not in message or "type" not in message:
        return None
    if message["type"] != "reasoning":
        return None
    if "status" in message and message["status"] not in (
        "in_progress",
        "completed",
        "incomplete",
    ):
        return None
    return message  # type: ignore


def return_response_image_generation_call(
    message: "OPENAI_MESSAGE_PARAM_TYPES",
) -> ImageGenerationCall | None:
    """Return ImageGenerationCall when the message is a valid image generation call."""
    if (
        "id" not in message
        or "result" not in message
        or "status" not in message
        or "type" not in message
    ):
        return None
    if message["type"] != "image_generation_call":
        return None
    if message["status"] not in ("in_progress", "completed", "generating", "failed"):
        return None
    return message  # type: ignore


def return_response_code_interpreter_tool_call(
    message: "OPENAI_MESSAGE_PARAM_TYPES",
) -> ResponseCodeInterpreterToolCallParam | None:
    """Return ResponseCodeInterpreterToolCallParam for a valid code-interpreter call."""
    if (
        "id" not in message
        or "code" not in message
        or "container_id" not in message
        or "outputs" not in message
        or "status" not in message
        or "type" not in message
    ):
        return None
    if message["type"] != "code_interpreter_call":
        return None
    if message["status"] not in (
        "in_progress",
        "completed",
        "incomplete",
        "interpreting",
        "failed",
    ):
        return None
    return message  # type: ignore


def return_response_local_shell_call(
    message: "OPENAI_MESSAGE_PARAM_TYPES",
) -> LocalShellCall | None:
    """Returns a LocalShellCall if the message is a valid local shell call."""
    if (
        "id" not in message
        or "action" not in message
        or "call_id" not in message
        or "status" not in message
        or "type" not in message
    ):
        return None
    if message["type"] != "local_shell_call":
        return None
    if message["status"] not in ("in_progress", "completed", "incomplete"):
        return None
    return message  # type: ignore


def return_response_local_shell_call_output(
    message: "OPENAI_MESSAGE_PARAM_TYPES",
) -> LocalShellCallOutput | None:
    """Return LocalShellCallOutput when the message is a valid local shell output."""
    if "id" not in message or "output" not in message or "type" not in message:
        return None
    if message["type"] != "local_shell_call_output":
        return None
    if "status" in message and message["status"] not in (
        "in_progress",
        "completed",
        "incomplete",
    ):
        return None
    return message  # type: ignore


def return_response_mcp_list_tools(
    message: "OPENAI_MESSAGE_PARAM_TYPES",
) -> McpListTools | None:
    """Return McpListTools when the message is a valid MCP list-tools object."""
    if (
        "id" not in message
        or "server_label" not in message
        or "tools" not in message
        or "type" not in message
    ):
        return None
    if message["type"] != "mcp_list_tools":
        return None
    return message  # type: ignore


def return_response_mcp_approval_request(
    message: "OPENAI_MESSAGE_PARAM_TYPES",
) -> McpApprovalRequest | None:
    """Return McpApprovalRequest when the message is a valid MCP approval request."""
    if (
        "id" not in message
        or "arguments" not in message
        or "name" not in message
        or "server_label" not in message
        or "type" not in message
    ):
        return None
    if message["type"] != "mcp_approval_request":
        return None
    return message  # type: ignore


def return_response_mcp_approval_response(
    message: "OPENAI_MESSAGE_PARAM_TYPES",
) -> McpApprovalResponse | None:
    """Returns an McpApprovalResponse if the message is a valid
    mcp approval response."""
    if (
        "approval_request_id" not in message
        or "approve" not in message
        or "type" not in message
    ):
        return None
    if message["type"] != "mcp_approval_response":
        return None
    return message  # type: ignore


def return_response_mcp_call(message: "OPENAI_MESSAGE_PARAM_TYPES") -> McpCall | None:
    """Return McpCall when the message is a valid MCP call."""
    if (
        "id" not in message
        or "arguments" not in message
        or "name" not in message
        or "server_label" not in message
        or "type" not in message
    ):
        return None
    if message["type"] != "mcp_call":
        return None
    return message  # type: ignore


def return_response_item_reference(
    message: "OPENAI_MESSAGE_PARAM_TYPES",
) -> ItemReference | None:
    """Return ItemReference when the message is a valid item reference."""
    if "id" not in message or "type" not in message:
        return None
    if message["type"] != "item_reference":
        return None
    return message  # type: ignore


def return_response_computer_tool_call_output_screenshot(
    message: "OPENAI_MESSAGE_PARAM_TYPES",
) -> ResponseComputerToolCallOutputScreenshotParam | None:
    """Return screenshot output param for a valid computer tool call output.
    Accepts either `file_id` or `image_url`.
    """
    if "type" not in message:
        return None
    if message["type"] != "computer_screenshot":
        return None
    if "file_id" not in message and "image_url" not in message:
        return None
    return message  # type: ignore


def return_chat_cmpl_tool_message(
    message: "OPENAI_MESSAGE_PARAM_TYPES",
) -> ChatCompletionToolMessageParam | None:
    """Returns ChatCompletionToolMessageParam if message is valid tool message.
    Validates required fields for tool messages.
    """
    if (
        "content" not in message
        or "role" not in message
        or "tool_call_id" not in message
    ):
        return None
    if message["role"] != "tool":
        return None
    return message  # type: ignore


def return_chat_cmpl_user_message(
    message: "OPENAI_MESSAGE_PARAM_TYPES",
) -> ChatCompletionUserMessageParam | None:
    """Returns ChatCompletionUserMessageParam if message is valid user message.
    Validates required fields for user messages.
    """
    if "content" not in message or "role" not in message:
        return None
    if message["role"] != "user":
        return None
    return message  # type: ignore


def return_chat_cmpl_system_message(
    message: "OPENAI_MESSAGE_PARAM_TYPES",
) -> ChatCompletionSystemMessageParam | None:
    """Return ChatCompletionSystemMessageParam for a valid system message."""
    if "content" not in message or "role" not in message:
        return None
    if message["role"] != "system":
        return None
    return message  # type: ignore


def return_chat_cmpl_function_message(
    message: "OPENAI_MESSAGE_PARAM_TYPES",
) -> ChatCompletionFunctionMessageParam | None:
    """Return ChatCompletionFunctionMessageParam for a valid function message."""
    if "content" not in message or "name" not in message or "role" not in message:
        return None
    if message["role"] != "function":
        return None
    return message  # type: ignore


def return_chat_cmpl_assistant_message(
    message: "OPENAI_MESSAGE_PARAM_TYPES",
) -> ChatCompletionAssistantMessageParam | None:
    """Returns ChatCompletionAssistantMessageParam if message is valid assistant.
    Validates required fields for assistant messages.
    """
    if "role" not in message:
        return None
    if message["role"] != "assistant":
        return None
    if (
        "content" not in message
        and "tool_calls" not in message
        and "function_call" not in message
    ):
        return None
    return message  # type: ignore


def return_chat_cmpl_developer_message(
    message: "OPENAI_MESSAGE_PARAM_TYPES",
) -> ChatCompletionDeveloperMessageParam | None:
    """Return ChatCompletionDeveloperMessageParam for a valid developer message."""
    if "content" not in message or "role" not in message:
        return None
    if message["role"] != "developer":
        return None
    return message  # type: ignore
