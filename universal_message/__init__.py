# universal_message/__init__.py
import datetime
import logging
import pathlib
import textwrap
import time
import typing
import zoneinfo

import agents
import durl
import jinja2
import pydantic
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)
from openai.types.chat.chat_completion_developer_message_param import (
    ChatCompletionDeveloperMessageParam,
)
from openai.types.chat.chat_completion_message import ChatCompletionMessage
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
from openai.types.responses.response_function_tool_call_param import (
    ResponseFunctionToolCallParam,
)
from openai.types.responses.response_input_item_param import (
    FunctionCallOutput,
    ResponseInputItemParam,
)
from rich.pretty import pretty_repr
from str_or_none import str_or_none

from ._id import generate_object_id

__version__ = pathlib.Path(__file__).parent.joinpath("VERSION").read_text().strip()

logger = logging.getLogger(__name__)

PRIMITIVE_TYPES: typing.TypeAlias = typing.Union[str, int, float, bool, None]

OPENAI_MESSAGE_PARAM_TYPES: typing.TypeAlias = typing.Union[
    ResponseInputItemParam,
    ChatCompletionMessageParam,
    typing.Dict,
]

SUPPORTED_MESSAGE_TYPES: typing.TypeAlias = typing.Union[
    "Message",
    str,
    durl.DataURL,
    ChatCompletionMessage,
    pydantic.BaseModel,
    OPENAI_MESSAGE_PARAM_TYPES,
]


class Message(pydantic.BaseModel):
    """A universal message format for AI interactions."""

    # Required fields
    role: typing.Literal["user", "assistant", "system", "developer", "tool"]
    """
    Role 'user' is for user input.
    Role 'assistant' is for assistant output or assistant tool call.
    Role 'system' is for system instructions.
    Role 'developer' is for developer output.
    Role 'tool' is for tool output.
    """

    content: str  # I love simple definitions
    """The field must be a plain text content or data URL"""

    channel: typing.Optional[typing.Literal["analysis", "commentary", "final"]] = None
    """
    Channel 'analysis' is for thinking or reasoning.
    Channel 'commentary' is for tool call or tool output.
    Channel 'final' is for final output of the assistant.
    """

    # Optional fields
    id: str = pydantic.Field(default_factory=generate_object_id)
    call_id: typing.Optional[str] = None
    tool_name: typing.Optional[str] = None
    arguments: typing.Optional[str] = None
    created_at: int = pydantic.Field(default_factory=lambda: int(time.time()))
    metadata: typing.Optional[typing.Dict[str, PRIMITIVE_TYPES]] = None

    @pydantic.model_validator(mode="after")
    def validate_content(self) -> typing.Self:
        if might_content := str_or_none(self.content):
            self.content = might_content
        else:
            if self.call_id and self.tool_name and self.arguments:  # Tool call
                self.content = (
                    f"[tool_call:{self.tool_name}](#{self.call_id}):{self.arguments}"
                )
            raise ValueError(f"Invalid content input: {self.content}")

        if self.content.startswith("data:") and not durl.DataURL.is_data_url(
            self.content
        ):
            logger.warning(
                "The content starts with 'data:' but is not a valid data URL: "
                + f"{pretty_repr(self.content, max_string=100)}"
            )
        return self

    @pydantic.model_validator(mode="after")
    def validate_channel(self) -> typing.Self:
        if not self.channel:
            if self.role == "assistant" and self.call_id is not None:
                self.channel = "commentary"
        return self

    @classmethod
    def from_any(cls, data: SUPPORTED_MESSAGE_TYPES) -> "Message":
        """Create message from various input types."""
        from universal_message.utils.any_to_message import from_any

        return from_any(data)

    @classmethod
    def from_text_of_oss(cls, text: str) -> typing.List["Message"]:
        from universal_message.utils.messages_from_text_of_oss import (
            messages_from_text_of_oss,
        )

        return messages_from_text_of_oss(text)

    def to_instructions(
        self,
        *,
        with_datetime: bool = False,
        tz: zoneinfo.ZoneInfo | str | None = None,
        max_string: int = 600,
    ) -> str:
        """Format message as readable instructions."""
        from universal_message.utils.ensure_tz import ensure_tz

        _role = self.role
        _content = self.content

        _dt: datetime.datetime | None = None
        if with_datetime:
            _dt = datetime.datetime.fromtimestamp(self.created_at, ensure_tz(tz))
            _dt = _dt.replace(microsecond=0)
        template = jinja2.Template(
            textwrap.dedent(
                """
                [{% if dt %}{{ dt.strftime('%Y-%m-%dT%H:%M:%S') }} {% endif %}{{ role }}] {{ content }}
                """  # noqa: E501
            ).strip()
        )
        return template.render(
            role=_role,
            dt=_dt,
            content=pretty_repr(_content, max_string=max_string),
        ).strip()

    def to_responses_input_item(self) -> ResponseInputItemParam:
        """Convert to OpenAI responses API format."""
        _role = self.role.lower()
        _content = self.content

        if _role in ("user", "assistant", "system", "developer"):
            if self.call_id or self.tool_name:
                if not self.call_id:
                    raise ValueError("Function tool call must have a call_id")
                if not self.tool_name:
                    raise ValueError("Function tool call must have a tool_name")
                _args = self.arguments or "{}"
                return ResponseFunctionToolCallParam(
                    arguments=_args,
                    call_id=self.call_id,
                    name=self.tool_name,
                    type="function_call",
                )
            else:
                return EasyInputMessageParam(role=_role, content=_content)
        elif _role in ("tool",):
            if not self.call_id:
                raise ValueError("Tool message must have a call_id")
            return FunctionCallOutput(
                call_id=self.call_id, output=_content, type="function_call_output"
            )
        else:
            logger.warning(f"Not supported role '{_role}' would be converted to 'user'")
            return EasyInputMessageParam(role="user", content=_content)

    def to_chat_cmpl_message(self) -> ChatCompletionMessageParam:
        """Convert to OpenAI chat completion format."""
        _role = self.role.lower()
        _content = self.content

        if _role in ("system",):
            return ChatCompletionSystemMessageParam(role=_role, content=_content)
        elif _role in ("developer",):
            return ChatCompletionDeveloperMessageParam(role=_role, content=_content)
        elif _role in ("user",):
            return ChatCompletionUserMessageParam(role=_role, content=_content)
        elif _role in ("assistant",):
            if self.call_id or self.tool_name:
                if not self.call_id:
                    raise ValueError("Assistant message must have a call_id")
                if not self.tool_name:
                    raise ValueError("Assistant message must have a tool_name")
                _tool = self.tool_name
                _args = self.arguments or "{}"
                return ChatCompletionAssistantMessageParam(
                    role=_role,
                    content=_content,
                    tool_calls=[
                        {
                            "id": self.call_id,
                            "function": {"name": _tool, "arguments": _args},
                            "type": "function",
                        }
                    ],
                )
            else:
                return ChatCompletionAssistantMessageParam(role=_role, content=_content)
        elif _role in ("tool",):
            if not self.call_id:
                raise ValueError("Tool message must have a call_id")
            return ChatCompletionToolMessageParam(
                role=_role, content=_content, tool_call_id=self.call_id
            )
        else:
            logger.warning(f"Not supported role '{_role}' would be converted to 'user'")
            return ChatCompletionUserMessageParam(role="user", content=_content)


def messages_from_any_items(
    items: (
        typing.List[SUPPORTED_MESSAGE_TYPES]
        | typing.List[Message]
        | typing.List[ResponseInputItemParam]
        | typing.List[ChatCompletionMessageParam]
        | typing.List[agents.TResponseInputItem]
        | typing.List[typing.Dict]
    ),
) -> typing.List[Message]:
    """Converts a list of items into a list of Message objects."""
    return [Message.from_any(item) for item in items]


def messages_to_instructions(
    messages: typing.List[Message],
    *,
    with_datetime: bool = False,
    tz: zoneinfo.ZoneInfo | str | None = None,
    sep: str = "\n\n",
) -> str:
    """Format multiple messages as readable instructions."""
    return sep.join(
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
