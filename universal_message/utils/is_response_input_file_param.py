import typing


def is_response_input_file_param(content: typing.Dict) -> bool:
    """Checks if content is a response input file parameter."""
    if "type" in content and content["type"] == "input_file":
        return True
    return False
