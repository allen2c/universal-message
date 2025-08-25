import logging
import typing

import durl
from openai.types.chat.chat_completion_content_part_param import (
    ChatCompletionContentPartParam,
)

logger = logging.getLogger(__name__)


def chat_cmpl_content_part_to_str(
    content: (
        str
        | ChatCompletionContentPartParam
        | typing.List[ChatCompletionContentPartParam]
    ),
) -> str:
    """Convert chat completion content parts to string format.
    Handles text, image, audio, and file content types.
    """

    def _chat_cmpl_content_part_to_str(
        content: ChatCompletionContentPartParam,
    ) -> list[str]:
        _contents: list[str] = []
        if content["type"] == "text":
            _contents.append(content["text"])
        elif content["type"] == "image_url":
            if durl.DataURL.is_data_url(content["image_url"]["url"]):
                _contents.append(
                    str(durl.DataURL.from_url(content["image_url"]["url"]))
                )
            else:
                _contents.append(content["image_url"]["url"])
        elif content["type"] == "input_audio":
            _format: typing.Literal["wav", "mp3"] = content["input_audio"]["format"]
            _mime_type = durl.MIMEType(
                "audio/wav" if _format == "wav" else "audio/mpeg"
            )
            _contents.append(
                str(
                    durl.DataURL(
                        mime_type=_mime_type, data=content["input_audio"]["data"]
                    )
                )
            )
        elif content["type"] == "file":
            if "file_data" in content["file"]:
                _contents.append(
                    str(
                        durl.DataURL(
                            mime_type=durl.MIMEType("text/plain"),
                            data=content["file"]["file_data"],
                        )
                    )
                )
            elif "file_id" in content["file"]:
                _contents.append(content["file"]["file_id"])
            else:
                logger.warning(f"Unhandled file content: {content}")
        else:
            raise ValueError(f"Invalid content type: {content['type']}")
        return _contents

    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        return "\n\n".join(
            ["\n\n".join(_chat_cmpl_content_part_to_str(_c)) for _c in content]
        ).strip()
    else:
        return "\n\n".join(_chat_cmpl_content_part_to_str(content)).strip()
