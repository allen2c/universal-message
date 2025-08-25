import typing


def is_response_input_message_content_list_param(
    content: typing.List[typing.Dict],
) -> bool:
    """Checks if content is a list of response input message content parts."""

    from universal_message.utils.is_response_input_file_param import (
        is_response_input_file_param,
    )
    from universal_message.utils.is_response_input_image_param import (
        is_response_input_image_param,
    )
    from universal_message.utils.is_response_input_text_param import (
        is_response_input_text_param,
    )

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
