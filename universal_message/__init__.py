import base64
import datetime
import pathlib
import textwrap
import time
import typing
import zoneinfo

import jinja2
import pydantic

from universal_message._id import generate_object_id

__version__ = pathlib.Path(__file__).parent.joinpath("VERSION").read_text().strip()

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


class DataURL(pydantic.BaseModel):
    """RFC 2397: `data:[<mediatype>][;<parameter>][;base64],<data>`

    e.g. data:text/plain;charset=UTF-8;base64,SGVsbG8=
    """

    mime_type: MIME_TYPE_TYPES
    parameters: str | None = None
    encoded: typing.Literal["base64"] = "base64"
    data: str

    @pydantic.model_validator(mode="after")
    def validate_parameters(self) -> typing.Self:
        if self.parameters is None:
            return self

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
    def from_data(
        cls,
        mime_type: MIME_TYPE_TYPES,
        raw_data: str | bytes,
        *,
        parameters: str | None = None,
    ) -> "DataURL":
        if isinstance(raw_data, str):
            data = base64.b64encode(raw_data.encode("utf-8")).decode("utf-8")
        else:
            data = base64.b64encode(raw_data).decode("utf-8")

        return cls(mime_type=mime_type, parameters=parameters, data=data)

    @property
    def url(self) -> str:
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
        if self.encoded == "base64":
            return base64.b64decode(self.data).decode("utf-8")
        return self.data


class Message(pydantic.BaseModel):
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

    def to_instructions(
        self, *, with_datetime: bool = False, tz: zoneinfo.ZoneInfo | str | None = None
    ) -> str:
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


def messages_to_instructions(
    messages: typing.List[Message],
    *,
    with_datetime: bool = False,
    tz: zoneinfo.ZoneInfo | str | None = None,
) -> str:
    return "\n\n".join(
        message.to_instructions(with_datetime=with_datetime, tz=tz)
        for message in messages
    )


def _ensure_tz(tz: zoneinfo.ZoneInfo | str | None) -> zoneinfo.ZoneInfo:
    if tz is None:
        return zoneinfo.ZoneInfo("UTC")
    if not isinstance(tz, zoneinfo.ZoneInfo):
        return zoneinfo.ZoneInfo(tz)
    return tz
