import typing


def is_response_output_text_param(content: typing.Dict) -> bool:
    """Checks if content is a response output text parameter."""
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
