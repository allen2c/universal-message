import json
import textwrap
import typing

from universal_message import Message
from universal_message.utils.messages_from_plaintext_of_gpt_oss import (
    messages_from_plaintext_of_gpt_oss,
)

Primitive = typing.Union[str, int, float, bool, None]


def test_messages_from_text_of_oss_parses_metadata_and_messages() -> None:
    """Validate docstring metadata and OSS message parsing."""
    sample_text: str = textwrap.dedent(
        '''
        """
        Customer requests warranty repair for vehicle making unusual noises.

        ROLES: user (describes issue), assistant (arranges service logistics)
        CHANNELS: analysis, commentary, final

        TOOLS:
        ```json
        [
          {
            "name": "verify_warranty_status",
            "description": "Verify warranty",
            "parameters": {
              "type": "object",
              "properties": {
                "vin": {"type": "string"},
                "as_of_date": {"type": "string", "format": "date"}
              },
              "required": ["vin", "as_of_date"]
            },
            "strict": true
          },
          {
            "name": "get_service_appointments",
            "description": "Find appointment slots",
            "parameters": {
              "type": "object",
              "properties": {
                "location_id": {"type": "string"},
                "service_type": {"type": "string"},
                "start_date": {"type": "string", "format": "date"},
                "end_date": {"type": "string", "format": "date"}
              },
              "required": [
                "location_id", "service_type", "start_date", "end_date"
              ]
            },
            "strict": true
          }
        ]
        ```
        """

        system:
        You are a service advisor at AutoCare dealership.

        user:
        My 2023 Honda Accord is making a grinding noise when I brake.
        It's under warranty—can you help?

        assistant channel=analysis:
        Ask for VIN and contact; then verify warranty and fetch slots.

        assistant channel=final:
        I'm sorry about the grinding noise—we'll get this handled.
        Could you share your VIN, mileage, preferred time, and phone number?

        user:
        VIN 1HG..., ~18,200 miles, tomorrow morning, +1555...

        assistant channel=commentary to=tool.verify_warranty_status:
        {"vin":"1HG...","as_of_date":"2025-08-23"}

        tool.verify_warranty_status channel=commentary to=assistant:
        {"status":"active"}

        assistant channel=commentary to=tool.get_service_appointments:
        {"location_id":"autocare_sfo_01","service_type":"warranty_repair",
         "start_date":"2025-08-23","end_date":"2025-08-24"}

        tool.get_service_appointments channel=commentary to=assistant:
        {"slots":[{"slot_id":"slot_2025-08-23_08:00"}]}

        assistant channel=analysis:
        Choose 8:00 AM slot, then confirm with the user.

        assistant channel=final:
        Warranty is active. I can hold the 8:00 AM slot tomorrow.
        Would you like me to book it and text a confirmation?
        '''
    ).strip()

    messages: typing.List[Message] = messages_from_plaintext_of_gpt_oss(
        text=sample_text
    )

    # Basic shape
    assert isinstance(messages, list)
    assert len(messages) == 11

    # Metadata injected on first message
    meta0 = messages[0].metadata
    assert isinstance(meta0, dict)
    meta: typing.Mapping[str, Primitive] = typing.cast(
        typing.Mapping[str, Primitive], meta0
    )
    assert "dialogue_description" in meta
    desc = typing.cast(str, meta["dialogue_description"])  # type: ignore[index]
    assert desc.startswith("Customer requests warranty repair")

    assert "dialogue_roles" in meta
    roles = typing.cast(str, meta["dialogue_roles"])  # type: ignore[index]
    assert "assistant" in roles

    assert "dialogue_channels" in meta
    channels = typing.cast(str, meta["dialogue_channels"])  # type: ignore[index]
    assert "analysis" in channels

    assert "dialogue_tools_definitions" in meta
    tools_json: str = typing.cast(
        str, meta["dialogue_tools_definitions"]  # type: ignore[index]
    )
    parsed_tools: typing.List[typing.Dict[str, object]] = json.loads(tools_json)
    tool_names: typing.List[str] = [typing.cast(str, t["name"]) for t in parsed_tools]
    assert "verify_warranty_status" in tool_names
    assert "get_service_appointments" in tool_names

    # Roles and channels
    assert messages[0].role == "system"
    assert messages[1].role == "user"
    assert messages[2].role == "assistant" and messages[2].channel == "analysis"
    assert messages[3].role == "assistant" and messages[3].channel == "final"

    # Assistant tool call
    call_msg: Message = messages[5]
    assert call_msg.role == "assistant"
    assert call_msg.channel == "commentary"
    assert call_msg.tool_name == "verify_warranty_status"
    assert isinstance(call_msg.arguments, str)
    call_args: typing.Dict[str, object] = json.loads(call_msg.arguments)
    assert call_args == {"vin": "1HG...", "as_of_date": "2025-08-23"}

    # Tool result
    tool_msg: Message = messages[6]
    assert tool_msg.role == "tool"
    assert tool_msg.channel == "commentary"
    assert tool_msg.tool_name == "verify_warranty_status"
    tool_payload: typing.Dict[str, object] = json.loads(tool_msg.content)
    assert tool_payload == {"status": "active"}

    # Second tool call and result
    call_msg_2: Message = messages[7]
    assert call_msg_2.role == "assistant"
    assert call_msg_2.channel == "commentary"
    assert call_msg_2.tool_name == "get_service_appointments"
    args2: typing.Dict[str, object] = json.loads(call_msg_2.arguments or "{}")
    assert args2["location_id"] == "autocare_sfo_01"
    assert args2["service_type"] == "warranty_repair"

    tool_msg_2: Message = messages[8]
    assert tool_msg_2.role == "tool"
    payload2: typing.Dict[str, object] = json.loads(tool_msg_2.content)
    assert payload2 == {"slots": [{"slot_id": "slot_2025-08-23_08:00"}]}

    # Final assistant reasoning and reply
    assert messages[9].role == "assistant" and messages[9].channel == "analysis"
    assert messages[10].role == "assistant" and messages[10].channel == "final"
