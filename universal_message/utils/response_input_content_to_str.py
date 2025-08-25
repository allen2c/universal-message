import logging
import typing

import durl
from openai.types.responses.response_input_content_param import (
    ResponseInputContentParam,
)
from openai.types.responses.response_input_message_content_list_param import (
    ResponseInputMessageContentListParam,
)

logger = logging.getLogger(__name__)


def response_input_content_to_str(
    content: typing.Union[
        str, ResponseInputContentParam, ResponseInputMessageContentListParam
    ],
) -> str:
    """Convert response input content to string format.
    Handles text, image, and file content types.
    """

    def _input_content_param_to_str(content: ResponseInputContentParam) -> str:
        _contents: typing.List[typing.Union[str, durl.DataURL]] = []
        for _c in content:
            if isinstance(_c, str):
                return _c
            if _c["type"] == "input_text":
                _contents.append(_c["text"])
            elif _c["type"] == "input_image":
                if _image_url := _c.get("image_url"):
                    if durl.DataURL.is_data_url(_image_url):
                        _contents.append(durl.DataURL.from_url(_image_url))
                    else:
                        _contents.append(_image_url)
                elif _file_id := _c.get("file_id"):
                    _contents.append(_file_id)
                else:
                    logger.warning(f"Unhandled image content: {_c}")
            elif _c["type"] == "input_file":
                if _file_data := _c.get("file_data"):
                    if durl.DataURL.is_data_url(_file_data):
                        _contents.append(durl.DataURL.from_url(_file_data))
                    else:
                        _contents.append(_file_data)
                elif _file_url := _c.get("file_url"):
                    if durl.DataURL.is_data_url(_file_url):
                        _contents.append(durl.DataURL.from_url(_file_url))
                    else:
                        _contents.append(_file_url)
                elif _file_id := _c.get("file_id"):
                    _contents.append(_file_id)
                else:
                    logger.warning(f"Unhandled file content: {_c}")
            else:
                logger.warning(f"Unhandled content: {_c}")
        return "\n\n".join([str(_c) for _c in _contents])

    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        return "\n\n".join([_input_content_param_to_str(_c) for _c in content])
    else:
        return _input_content_param_to_str(content)
