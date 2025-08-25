import typing


def is_response_input_text_param(content: typing.Dict) -> bool:
    """Checks if content is a response input text parameter."""
    if "type" in content and content["type"] == "input_text":
        return True
    return False
