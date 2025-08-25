import typing


def is_response_input_image_param(content: typing.Dict) -> bool:
    """Checks if content is a response input image parameter."""
    if "type" in content and content["type"] == "input_image":
        return True
    return False
