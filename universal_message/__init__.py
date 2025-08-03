import base64
import datetime
import logging
import pathlib
import re
import textwrap
import time
import typing
import urllib.parse
import zoneinfo

import jinja2
import pydantic
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)
from openai.types.chat.chat_completion_developer_message_param import (
    ChatCompletionDeveloperMessageParam,
)
from openai.types.chat.chat_completion_function_message_param import (
    ChatCompletionFunctionMessageParam,
)
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
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
from openai.types.responses.response_input_content_param import (
    ResponseInputContentParam,
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
from openai.types.responses.response_input_item_param import (
    ResponseInputItemParam,
)
from openai.types.responses.response_input_message_content_list_param import (
    ResponseInputMessageContentListParam,
)
from openai.types.responses.response_output_message_param import (
    ResponseOutputMessageParam,
)
from openai.types.responses.response_reasoning_item_param import (
    ResponseReasoningItemParam,
)
from rich.pretty import pretty_repr

from universal_message._id import generate_object_id

__version__ = pathlib.Path(__file__).parent.joinpath("VERSION").read_text().strip()

logger = logging.getLogger(__name__)

PRIMITIVE_TYPES: typing.TypeAlias = typing.Union[str, int, float, bool, None]
MIME_TYPE_TYPES: typing.TypeAlias = (
    typing.Literal[
        "text/plain",
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/webp",
        "image/svg+xml",
        "audio/mpeg",
        "audio/wav",
        "audio/webm",
    ]
    | str
)
OPENAI_MESSAGE_PARAM_TYPES: typing.TypeAlias = typing.Union[
    ResponseInputItemParam,
    ChatCompletionMessageParam,
    typing.Dict,
]


DATA_URL_PATTERN = re.compile(
    r"""
    ^data:
    (?P<mediatype>[^;,]*)  # optional MIME type
    (?:  # whole parameter section
        ;  # ← semicolon stays here
        (?P<params>
            [^;,=]+=[^;,]*  # first attr=value
            (?:;[^;,=]+=[^;,]*)*  # 0-n more ;attr=value
        )
    )?  # entire param list is optional
    (?P<base64>;base64)?  # optional ;base64 flag
    ,
    (?P<payload>.*)  # everything after the first comma
    \Z
    """,
    re.I | re.S | re.VERBOSE,
)


class DataURL(pydantic.BaseModel):
    """Data URL representation per RFC 2397.

    Format: data:[<mediatype>][;<parameter>][;base64],<data>
    Example: data:text/plain;charset=UTF-8;base64,SGVsbG8=
    """

    mime_type: MIME_TYPE_TYPES
    parameters: str | None = None
    encoded: typing.Literal["base64"] | None = "base64"
    data: str

    @pydantic.model_validator(mode="after")
    def validate_parameters(self) -> typing.Self:
        if self.parameters is None:
            return self

        if self.parameters.startswith(";"):
            self.parameters = self.parameters.lstrip(";")

        parts = self.parameters.split(";")
        for part in parts:
            if not part:
                continue
            if "=" not in part:
                raise ValueError(f"Invalid parameter format for '{part}': missing '='")
            key, value = part.split("=", 1)
            if not key.strip() or not value.strip():
                raise ValueError(
                    f"Invalid parameter format for '{part}': empty key or value"
                )
        return self

    @pydantic.model_serializer
    def serialize_model(self) -> str:
        return self.url

    @classmethod
    def is_data_url(cls, url: str) -> bool:
        """Check if URL is a valid data URL."""
        return bool(DATA_URL_PATTERN.match(url))

    @classmethod
    def from_url(cls, url: str) -> "DataURL":
        """Create DataURL from URL string."""
        mime_type, parameters, encoded, data = cls.__parse_url(url)
        return cls(
            mime_type=mime_type, parameters=parameters, encoded=encoded, data=data
        )

    @classmethod
    def from_data(
        cls,
        mime_type: MIME_TYPE_TYPES,
        raw_data: str | bytes,
        *,
        parameters: str | None = None,
    ) -> "DataURL":
        """Create DataURL from raw data and MIME type."""
        if isinstance(raw_data, str):
            data = base64.b64encode(raw_data.encode("utf-8")).decode("utf-8")
        else:
            data = base64.b64encode(raw_data).decode("utf-8")

        return cls(mime_type=mime_type, parameters=parameters, data=data)

    @property
    def url(self) -> str:
        """Get the complete data URL string."""
        STRING_PATTERN = (
            "data:{mediatype}{might_semicolon_parameters}{semicolon_encoded},{data}"
        )

        return STRING_PATTERN.format(
            mediatype=self.mime_type,
            might_semicolon_parameters=f";{self.parameters}" if self.parameters else "",
            semicolon_encoded=f";{self.encoded}" if self.encoded else "",
            data=self.data,
        )

    @property
    def decoded_data(self) -> str:
        """Get decoded data payload."""
        if self.encoded == "base64":
            return base64.b64decode(self.data).decode("utf-8")
        return self.data

    @classmethod
    def __parse_url(cls, url: str) -> typing.Tuple[
        str,
        str | None,
        typing.Literal["base64"] | None,
        str,
    ]:
        """Parse data URL into components.

        Returns: (mime_type, parameters, encoded, data)
        Raises: ValueError if URL is invalid.
        """
        m = DATA_URL_PATTERN.match(url)
        if not m:
            raise ValueError("Not a valid data URL")

        mime_type: str = m.group("mediatype") or "text/plain"

        params: str | None = m.group("params")
        try:
            urllib.parse.parse_qsl(params)
        except ValueError as e:
            logger.warning(f"Invalid parameters in data URL, ignored: {pretty_repr(e)}")
            params = None

        encoded = bool(m.group("base64"))

        raw: str = m.group("payload")

        return (mime_type, params, "base64" if encoded else None, raw)


MESSAGE_CONTENT_SIMPLE_TYPES: typing.TypeAlias = typing.Union[
    str, DataURL, pydantic.HttpUrl
]
MESSAGE_CONTENT_LIST_TYPES: typing.TypeAlias = typing.List[MESSAGE_CONTENT_SIMPLE_TYPES]
MESSAGE_CONTENT_TYPES: typing.TypeAlias = typing.Union[
    MESSAGE_CONTENT_SIMPLE_TYPES,
    MESSAGE_CONTENT_LIST_TYPES,
]


class Message(pydantic.BaseModel):
    """Universal message format for AI interactions.

    Supports text, data URLs, HTTP URLs, or lists of content.
    """

    # Required fields
    role: typing.Literal["user", "assistant", "system", "developer", "tool"] | str
    content: MESSAGE_CONTENT_TYPES

    # Optional fields
    id: str = pydantic.Field(default_factory=generate_object_id)
    call_id: typing.Optional[str] = None
    created_at: int = pydantic.Field(default_factory=lambda: int(time.time()))
    metadata: typing.Optional[typing.Dict[str, PRIMITIVE_TYPES]] = None

    @classmethod
    def from_any(
        cls,
        data: (
            str
            | DataURL
            | pydantic.HttpUrl
            | pydantic.BaseModel
            | OPENAI_MESSAGE_PARAM_TYPES
        ),
    ) -> "Message":
        """Create message from various input types."""
        if isinstance(data, str):
            return Message(role="user", content=data)
        if isinstance(data, DataURL):
            return Message(role="user", content=data)
        if isinstance(data, pydantic.HttpUrl):
            return Message(role="user", content=data)
        if isinstance(data, pydantic.BaseModel):
            return cls.model_validate_json(data.model_dump_json())
        if m := return_response_easy_input_message(data):
            _content = (
                content_from_response_input_content_list_param(m["content"])
                if isinstance(m["content"], list)
                else content_from_response_input_content_param(m["content"])
            )
            return cls.model_validate({"role": m["role"], "content": _content})
        if m := return_response_input_message(data):
            raise NotImplementedError()
        if m := return_response_output_message(data):
            raise NotImplementedError()
        if m := return_response_file_search_tool_call(data):
            raise NotImplementedError()
        if m := return_response_computer_tool_call(data):
            raise NotImplementedError()
        if m := return_response_computer_call_output(data):
            raise NotImplementedError()
        if m := return_response_function_web_search(data):
            raise NotImplementedError()
        if m := return_response_function_tool_call(data):
            raise NotImplementedError()
        if m := return_response_function_call_output(data):
            raise NotImplementedError()
        if m := return_response_reasoning_item(data):
            raise NotImplementedError()
        if m := return_response_image_generation_call(data):
            raise NotImplementedError()
        if m := return_response_code_interpreter_tool_call(data):
            raise NotImplementedError()
        if m := return_response_local_shell_call(data):
            raise NotImplementedError()
        if m := return_response_local_shell_call_output(data):
            raise NotImplementedError()
        if m := return_response_mcp_list_tools(data):
            raise NotImplementedError()
        if m := return_response_mcp_approval_request(data):
            raise NotImplementedError()
        if m := return_response_mcp_approval_response(data):
            raise NotImplementedError()
        if m := return_response_mcp_call(data):
            raise NotImplementedError()
        if m := return_response_item_reference(data):
            raise NotImplementedError()
        if m := return_chat_cmpl_tool_message(data):
            raise NotImplementedError()
        if m := return_chat_cmpl_user_message(data):
            raise NotImplementedError()
        if m := return_chat_cmpl_system_message(data):
            raise NotImplementedError()
        if m := return_chat_cmpl_function_message(data):
            raise NotImplementedError()
        if m := return_chat_cmpl_assistant_message(data):
            raise NotImplementedError()
        if m := return_chat_cmpl_developer_message(data):
            raise NotImplementedError()

        return cls.model_validate(data)

    def to_instructions(
        self, *, with_datetime: bool = False, tz: zoneinfo.ZoneInfo | str | None = None
    ) -> str:
        """Format message as readable instructions."""
        _role = self.role
        _content = ""
        _dt: datetime.datetime | None = None
        if with_datetime:
            _dt = datetime.datetime.fromtimestamp(self.created_at, _ensure_tz(tz))
            _dt = _dt.replace(microsecond=0)
        template = jinja2.Template(
            textwrap.dedent(
                """
                {{ role }}:
                {% if datetime %}[{{ datetime }}] {% endif %}{{ content }}
                """
            ).strip()
        )
        return template.render(
            role=_role,
            datetime=_dt.isoformat() if _dt else None,
            content=_content,
        ).strip()

    def to_responses_input_item(self) -> ResponseInputItemParam:
        """Convert to OpenAI responses API format."""
        raise NotImplementedError("Not implemented")

    def to_chat_cmpl_message(self) -> ChatCompletionMessageParam:
        """Convert to OpenAI chat completion format."""
        raise NotImplementedError("Not implemented")


def messages_to_instructions(
    messages: typing.List[Message],
    *,
    with_datetime: bool = False,
    tz: zoneinfo.ZoneInfo | str | None = None,
) -> str:
    """Format multiple messages as readable instructions."""
    return "\n\n".join(
        message.to_instructions(with_datetime=with_datetime, tz=tz)
        for message in messages
    )


def messages_to_responses_input_items(
    messages: typing.List[Message],
) -> typing.List[ResponseInputItemParam]:
    """Convert messages to OpenAI responses API format."""
    return [message.to_responses_input_item() for message in messages]


def messages_to_chat_cmpl_messages(
    messages: typing.List[Message],
) -> typing.List[ChatCompletionMessageParam]:
    """Convert messages to OpenAI chat completion format."""
    return [message.to_chat_cmpl_message() for message in messages]


def is_response_input_message_content_list_param(
    content: typing.List[typing.Dict],
) -> bool:
    if len(content) == 0:
        return False  # Empty list, invalid message content
    if any(
        is_response_input_file_param(item)
        or is_response_input_text_param(item)
        or is_response_input_image_param(item)
        for item in content
    ):
        return True
    return False


def is_response_input_file_param(content: typing.Dict) -> bool:
    if "type" in content and content["type"] == "input_file":
        return True
    return False


def is_response_input_text_param(content: typing.Dict) -> bool:
    if "type" in content and content["type"] == "input_text":
        return True
    return False


def is_response_input_image_param(content: typing.Dict) -> bool:
    if "type" in content and content["type"] == "input_image":
        return True
    return False


def is_response_output_text_param(content: typing.Dict) -> bool:
    if (
        "annotations" in content
        and "text" in content
        and "type" in content
        and content["type"] == "output_text"
        and isinstance(content["annotations"], list)
        and all("type" in item for item in content["annotations"])
        and any(
            item["type"]
            in ("file_citation", "url_citation", "container_file_citation", "file_path")
            for item in content["annotations"]
        )
    ):
        return True
    return False


def return_response_easy_input_message(
    message: OPENAI_MESSAGE_PARAM_TYPES,
) -> EasyInputMessageParam | None:
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
    message: OPENAI_MESSAGE_PARAM_TYPES,
) -> ResponseInputMessageParam | None:
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
    message: OPENAI_MESSAGE_PARAM_TYPES,
) -> ResponseOutputMessageParam | None:
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
    message: OPENAI_MESSAGE_PARAM_TYPES,
) -> ResponseFileSearchToolCallParam | None:
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
    message: OPENAI_MESSAGE_PARAM_TYPES,
) -> ResponseComputerToolCallParam | None:
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
    message: OPENAI_MESSAGE_PARAM_TYPES,
) -> ComputerCallOutput | None:
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
    message: OPENAI_MESSAGE_PARAM_TYPES,
) -> ResponseFunctionWebSearchParam | None:
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
    message: OPENAI_MESSAGE_PARAM_TYPES,
) -> ResponseFunctionToolCallParam | None:
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
    message: OPENAI_MESSAGE_PARAM_TYPES,
) -> FunctionCallOutput | None:
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
    message: OPENAI_MESSAGE_PARAM_TYPES,
) -> ResponseReasoningItemParam | None:
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
    message: OPENAI_MESSAGE_PARAM_TYPES,
) -> ImageGenerationCall | None:
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
    message: OPENAI_MESSAGE_PARAM_TYPES,
) -> ResponseCodeInterpreterToolCallParam | None:
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
    message: OPENAI_MESSAGE_PARAM_TYPES,
) -> LocalShellCall | None:
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
    message: OPENAI_MESSAGE_PARAM_TYPES,
) -> LocalShellCallOutput | None:
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
    message: OPENAI_MESSAGE_PARAM_TYPES,
) -> McpListTools | None:
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
    message: OPENAI_MESSAGE_PARAM_TYPES,
) -> McpApprovalRequest | None:
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
    message: OPENAI_MESSAGE_PARAM_TYPES,
) -> McpApprovalResponse | None:
    if (
        "approval_request_id" not in message
        or "approve" not in message
        or "type" not in message
    ):
        return None
    if message["type"] != "mcp_approval_response":
        return None
    return message  # type: ignore


def return_response_mcp_call(message: OPENAI_MESSAGE_PARAM_TYPES) -> McpCall | None:
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
    message: OPENAI_MESSAGE_PARAM_TYPES,
) -> ItemReference | None:
    if "id" not in message or "type" not in message:
        return None
    if message["type"] != "item_reference":
        return None
    return message  # type: ignore


def return_response_input_message_content_list(
    message: OPENAI_MESSAGE_PARAM_TYPES,
) -> ResponseInputMessageContentListParam | None:
    raise NotImplementedError()


def return_response_computer_tool_call_output_screenshot(
    message: OPENAI_MESSAGE_PARAM_TYPES,
) -> ResponseComputerToolCallOutputScreenshotParam | None:
    if "type" not in message:
        return None
    if message["type"] != "computer_screenshot":
        return None
    if "file_id" not in message and "image_url" not in message:
        return None
    return message  # type: ignore


def return_chat_cmpl_tool_message(
    message: OPENAI_MESSAGE_PARAM_TYPES,
) -> ChatCompletionToolMessageParam | None:
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
    message: OPENAI_MESSAGE_PARAM_TYPES,
) -> ChatCompletionUserMessageParam | None:
    if "content" not in message or "role" not in message:
        return None
    if message["role"] != "user":
        return None
    return message  # type: ignore


def return_chat_cmpl_system_message(
    message: OPENAI_MESSAGE_PARAM_TYPES,
) -> ChatCompletionSystemMessageParam | None:
    if "content" not in message or "role" not in message:
        return None
    if message["role"] != "system":
        return None
    return message  # type: ignore


def return_chat_cmpl_function_message(
    message: OPENAI_MESSAGE_PARAM_TYPES,
) -> ChatCompletionFunctionMessageParam | None:
    if "content" not in message or "name" not in message or "role" not in message:
        return None
    if message["role"] != "function":
        return None
    return message  # type: ignore


def return_chat_cmpl_assistant_message(
    message: OPENAI_MESSAGE_PARAM_TYPES,
) -> ChatCompletionAssistantMessageParam | None:
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
    message: OPENAI_MESSAGE_PARAM_TYPES,
) -> ChatCompletionDeveloperMessageParam | None:
    if "content" not in message or "role" not in message:
        return None
    if message["role"] != "developer":
        return None
    return message  # type: ignore


def content_from_response_input_content_param(
    content: str | ResponseInputContentParam,
) -> MESSAGE_CONTENT_SIMPLE_TYPES:
    if isinstance(content, str):
        return content
    if content["type"] == "input_text":
        return content["text"]
    elif content["type"] == "input_image":
        return pretty_repr(
            content.get("file_id") or content.get("image_url"), max_string=127
        ).strip("'\"")
    elif content["type"] == "input_file":
        return pretty_repr(
            content.get("file_id")
            or content.get("file_url")
            or content.get("file_data"),
            max_string=127,
        ).strip("'\"")
    else:
        raise ValueError(f"Invalid content type: {content['type']}")


def content_from_response_input_content_list_param(
    content: ResponseInputMessageContentListParam,
) -> MESSAGE_CONTENT_LIST_TYPES:
    return [content_from_response_input_content_param(item) for item in content]


def _ensure_tz(tz: zoneinfo.ZoneInfo | str | None) -> zoneinfo.ZoneInfo:
    """Ensure timezone is ZoneInfo object."""
    if tz is None:
        return zoneinfo.ZoneInfo("UTC")
    if not isinstance(tz, zoneinfo.ZoneInfo):
        return zoneinfo.ZoneInfo(tz)
    return tz
