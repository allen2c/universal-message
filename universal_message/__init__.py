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
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.responses import ResponseInputItemParam
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


class Message(pydantic.BaseModel):
    """Universal message format for AI interactions.

    Supports text, data URLs, HTTP URLs, or lists of content.
    """

    # Required fields
    role: typing.Literal["user", "assistant", "system", "developer", "tool"] | str
    content: typing.Union[
        str,
        DataURL,
        pydantic.HttpUrl,
        typing.List[typing.Union[str, DataURL, pydantic.HttpUrl]],
    ]

    # Optional fields
    id: str = pydantic.Field(default_factory=generate_object_id)
    call_id: typing.Optional[str] = None
    created_at: int = pydantic.Field(default_factory=lambda: int(time.time()))
    metadata: typing.Optional[typing.Dict[str, PRIMITIVE_TYPES]] = None

    def from_any(
        self,
        data: (
            str
            | DataURL
            | pydantic.HttpUrl
            | pydantic.BaseModel
            | ResponseInputItemParam
            | ChatCompletionMessageParam
            | typing.Dict
        ),
    ) -> "Message":
        """Create message from various input types."""
        raise NotImplementedError("Not implemented")

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


def _ensure_tz(tz: zoneinfo.ZoneInfo | str | None) -> zoneinfo.ZoneInfo:
    """Ensure timezone is ZoneInfo object."""
    if tz is None:
        return zoneinfo.ZoneInfo("UTC")
    if not isinstance(tz, zoneinfo.ZoneInfo):
        return zoneinfo.ZoneInfo(tz)
    return tz
