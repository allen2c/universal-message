# universal_message/utils/messages_from_text_of_oss.py
import re
import textwrap
import typing

import json_repair
import pydantic
from openai.types.shared.function_definition import FunctionDefinition

if typing.TYPE_CHECKING:
    from universal_message import Message


PRIMITIVE_TYPES: typing.TypeAlias = typing.Union[str, int, float, bool, None]

ListFunctionDefinitionAdapter = pydantic.TypeAdapter(typing.List[FunctionDefinition])


class OSSMetadata(typing.TypedDict, total=False):
    """Structured metadata extracted from the docstring header."""

    dialogue_description: str
    dialogue_roles: str  # Any string
    dialogue_channels: str  # Any string
    dialogue_tools_definitions: str  # JSON string


def messages_from_plaintext_of_gpt_oss(text: str) -> typing.List["Message"]:
    """Create messages from text of OSS."""
    from universal_message import Message

    text = text.strip()
    if not text:
        raise ValueError("Invalid text: empty text")

    # --- Parse optional docstring metadata block wrapped in triple quotes ---
    metadata: OSSMetadata = {}

    # Capture the first triple-quoted block if present
    docstring_match = re.match(r"^\s*\"\"\"([\s\S]*?)\"\"\"\s*\n?", text)
    docstring_text: str | None = None
    remaining_text = text
    if docstring_match:
        docstring_text = docstring_match.group(1).strip()
        remaining_text = text[docstring_match.end() :]

    if docstring_text:
        # 1) Dialogue description: first non-empty line
        lines = [ln.strip() for ln in docstring_text.splitlines()]
        description_lines: typing.List[str] = []
        i = 0
        while i < len(lines) and lines[i] == "":
            i += 1
        # Gather contiguous non-empty description lines until a known section starts
        while (
            i < len(lines)
            and not lines[i].startswith("ROLES:")
            and not lines[i].startswith("CHANNELS:")
            and not lines[i].startswith("TOOLS:")
        ):
            description_lines.append(lines[i])
            i += 1
        if description_lines:
            metadata["dialogue_description"] = " ".join(description_lines)

        # 2) ROLES line
        roles_match = re.search(r"^\s*ROLES:\s*(.*)$", docstring_text, re.MULTILINE)
        if roles_match:
            metadata["dialogue_roles"] = roles_match.group(1)

        # 3) CHANNELS line
        channels_match = re.search(
            r"^\s*CHANNELS:\s*(.*)$", docstring_text, re.MULTILINE
        )
        if channels_match:
            metadata["dialogue_channels"] = channels_match.group(1)

        # 4) TOOLS json code block
        tools_match = re.search(r"```json\s*\n([\s\S]*?)\n```", docstring_text)
        if tools_match:
            json_text = tools_match.group(1).strip()
            try:
                tools_raw = json_repair.loads(json_text)
                if isinstance(tools_raw, list):
                    # Validate and coerce to FunctionDefinition via pydantic model
                    tools = ListFunctionDefinitionAdapter.validate_python(
                        json_repair.loads(json_text)
                    )
                else:
                    tools = [FunctionDefinition.model_validate(tools_raw)]
                metadata["dialogue_tools_definitions"] = (
                    ListFunctionDefinitionAdapter.dump_json(tools).decode("utf-8")
                )
            except Exception:
                # Ignore malformed json tools block
                pass

    # --- Parse message sections in remaining text ---
    messages: typing.List[Message] = []

    # Accepted headers: system:, user:, assistant channel=... [to=tool.X]:
    # and tool.X channel=commentary to=assistant:
    header_re = re.compile(
        (
            r"^(system|user|assistant|tool(?:\.[A-Za-z0-9_\-]+)?)"  # role or tool.name
            r"(?:\s+channel=(analysis|commentary|final))?"  # optional channel
            r"(?:\s+to=(assistant|tool(?:\.[A-Za-z0-9_\-]+)?):|:)$"  # optional target
        )
    )

    current_role: str | None = None
    current_channel: str | None = None
    current_tool_name: str | None = None
    current_to_assistant: bool = False
    current_lines: typing.List[str] = []
    current_assistant_tool_name: str | None = None

    def flush_current() -> None:
        nonlocal current_role, current_channel, current_tool_name, current_lines
        nonlocal current_to_assistant, current_assistant_tool_name
        if current_role is None:
            return
        content = "\n".join(current_lines).rstrip()
        if current_role.startswith("tool."):
            # tool.<name> result must be commentary to assistant
            role = "tool"
            message = Message(role=role, content=content)
            # We keep tool name in metadata for downstream usage
            message.tool_name = current_tool_name
            message.channel = "commentary"
            messages.append(message)
        else:
            role = typing.cast(
                typing.Literal["user", "assistant", "system", "developer", "tool"],
                current_role,
            )
            message = Message(role=role, content=content)
            if current_channel:
                message.channel = typing.cast(
                    typing.Literal["analysis", "commentary", "final"],
                    current_channel,
                )
            if role == "assistant" and current_assistant_tool_name:
                # Treat as a tool call with arguments in content
                message.tool_name = current_assistant_tool_name
                message.arguments = content or "{}"
            messages.append(message)
        # reset
        current_role = None
        current_channel = None
        current_tool_name = None
        current_to_assistant = False
        current_lines = []
        current_assistant_tool_name = None

    for raw_line in remaining_text.splitlines():
        line = raw_line.rstrip()
        m = header_re.match(line)
        if m:
            # Flush previous section
            flush_current()
            hdr = m.group(1)
            ch = m.group(2)
            target = m.group(3)
            current_to_assistant = target == "assistant"
            if hdr.startswith("tool."):
                current_role = hdr  # temporary keep tool.<name>
                current_tool_name = hdr.split(".", 1)[1]
                current_channel = "commentary"
            else:
                current_role = hdr
                current_channel = ch
                # assistant tool call header: to=tool.NAME
                if (
                    current_role == "assistant"
                    and target
                    and target.startswith("tool.")
                ):
                    current_assistant_tool_name = target.split(".", 1)[1]
            continue
        # Content line
        current_lines.append(raw_line)

    # Flush tail
    flush_current()

    # Inject metadata (primitive-only) into first message if exists
    if messages:
        primitive_meta: typing.Dict[str, typing.Union[str, int, float, bool, None]]
        primitive_meta = {}
        if "dialogue_description" in metadata:
            primitive_meta["dialogue_description"] = typing.cast(
                str, metadata["dialogue_description"]
            )
        if "dialogue_roles" in metadata:
            primitive_meta["dialogue_roles"] = metadata["dialogue_roles"]
        if "dialogue_channels" in metadata:
            primitive_meta["dialogue_channels"] = metadata["dialogue_channels"]
        if "dialogue_tools_definitions" in metadata:
            primitive_meta["dialogue_tools_definitions"] = metadata[
                "dialogue_tools_definitions"
            ]
        messages[0].metadata = primitive_meta
    return messages


if __name__ == "__main__":
    text = textwrap.dedent(
        """
        \"\"\"
        Customer requests warranty repair for vehicle making unusual noises.
        ROLES: user (customer describing brake grinding on a 2023 Honda Accord), assistant (service advisor arranging warranty service and logistics)
        CHANNELS: analysis, commentary, final

        TOOLS:
        ```json
        [
        {
            "name": "verify_warranty_status",
            "description": "Verify whether a vehicle is under manufacturer warranty and return coverage details.",
            "parameters": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "vin": {
                "type": "string",
                "description": "17-character Vehicle Identification Number.",
                "pattern": "^[A-HJ-NPR-Z0-9]{17}$"
                },
                "odometer_miles": {
                "type": "integer",
                "minimum": 0,
                "description": "Current odometer reading in miles."
                },
                "as_of_date": {
                "type": "string",
                "format": "date",
                "description": "Date to evaluate warranty status (YYYY-MM-DD)."
                }
            },
            "required": ["vin", "as_of_date"]
            },
            "strict": true
        },
        {
            "name": "get_service_appointments",
            "description": "Retrieve available service appointment slots for a specific location and service type.",
            "parameters": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "location_id": {
                "type": "string",
                "description": "Dealership or service center identifier."
                },
                "service_type": {
                "type": "string",
                "enum": ["warranty_repair", "brake_inspection", "diagnostic"],
                "description": "Type of service appointment requested."
                },
                "start_date": {
                "type": "string",
                "format": "date",
                "description": "Start date for the search window (YYYY-MM-DD)."
                },
                "end_date": {
                "type": "string",
                "format": "date",
                "description": "End date for the search window (YYYY-MM-DD)."
                },
                "preferred_time_of_day": {
                "type": "string",
                "enum": ["morning", "afternoon", "evening"],
                "description": "Preferred time of day for the appointment."
                },
                "loaner_required": {
                "type": "boolean",
                "description": "Whether a loaner vehicle is requested."
                }
            },
            "required": ["location_id", "service_type", "start_date", "end_date"]
            },
            "strict": true
        },
        {
            "name": "create_service_ticket",
            "description": "Create a service ticket for a vehicle concern with optional appointment slot selection.",
            "parameters": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "customer_id": {
                "type": "string",
                "description": "Internal customer identifier."
                },
                "vin": {
                "type": "string",
                "description": "17-character Vehicle Identification Number.",
                "pattern": "^[A-HJ-NPR-Z0-9]{17}$"
                },
                "concern_description": {
                "type": "string",
                "description": "Customer-described primary concern for service.",
                "minLength": 3
                },
                "warranty_claim": {
                "type": "boolean",
                "description": "Whether this ticket should be submitted as a warranty claim."
                },
                "appointment_slot_id": {
                "type": "string",
                "description": "Identifier of the selected appointment slot, if already chosen."
                },
                "contact_phone_e164": {
                "type": "string",
                "description": "Customer phone number in E.164 format.",
                "pattern": "^\\+?[1-9]\\d{1,14}$"
                }
            },
            "required": ["customer_id", "vin", "concern_description", "warranty_claim"]
            },
            "strict": true
        },
        {
            "name": "request_loaner_vehicle",
            "description": "Reserve a loaner vehicle for an upcoming service appointment if available.",
            "parameters": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "location_id": {
                "type": "string",
                "description": "Dealership or service center identifier."
                },
                "date": {
                "type": "string",
                "format": "date",
                "description": "Date of the appointment (YYYY-MM-DD)."
                },
                "duration_hours": {
                "type": "integer",
                "minimum": 1,
                "maximum": 72,
                "description": "Expected loaner duration in hours."
                },
                "customer_id": {
                "type": "string",
                "description": "Internal customer identifier."
                },
                "vin": {
                "type": "string",
                "description": "17-character Vehicle Identification Number.",
                "pattern": "^[A-HJ-NPR-Z0-9]{17}$"
                }
            },
            "required": ["location_id", "date", "duration_hours", "customer_id", "vin"]
            },
            "strict": true
        },
        {
            "name": "send_confirmation_sms",
            "description": "Send a confirmation SMS to the customer with appointment and ticket details.",
            "parameters": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "phone_e164": {
                "type": "string",
                "description": "Destination phone number in E.164 format.",
                "pattern": "^\\+?[1-9]\\d{1,14}$"
                },
                "message": {
                "type": "string",
                "description": "SMS message body.",
                "minLength": 1,
                "maxLength": 480
                },
                "ticket_id": {
                "type": "string",
                "description": "Related service ticket identifier."
                }
            },
            "required": ["phone_e164", "message"]
            },
            "strict": true
        }
        ]
        ```
        \"\"\"

        system:
        You are a service advisor at AutoCare dealership. Handle warranty claims professionally, clarify missing details, and arrange convenient service appointments. Use tools only when necessary to verify warranty, find slots, open tickets, and send confirmations.

        user:
        My 2023 Honda Accord is making a grinding noise when I brake, and it's still under warranty. I need to get this fixed ASAP - can you help?

        assistant channel=analysis:
        Express empathy. We need VIN and contact details before calling tools. Start with clarification without calling tools. Offer safety guidance.

        assistant channel=final:
        I'm sorry about the grinding noise—that's frustrating and concerning. We'll take care of this promptly.

        Before I book anything, could you share:
        - VIN (17 characters)
        - Approximate mileage
        - Preferred day/time window (tomorrow morning/afternoon, or Thursday)
        - Best contact phone number (for confirmations)
        - Do you need a loaner vehicle?

        For safety, please minimize heavy braking and high speeds until we inspect the brakes.

        user:
        Sure. VIN: 1HGCV1F32PA123456, mileage ~18,200 miles. Tomorrow morning is good. Phone: +15551234567. A loaner would be helpful.

        assistant channel=analysis:
        We can now verify warranty and fetch morning slots for tomorrow. Then create a ticket, request a loaner, and send an SMS confirmation.

        assistant channel=commentary to=tool.verify_warranty_status:
        {"vin":"1HGCV1F32PA123456","odometer_miles":18200,"as_of_date":"2025-08-23"}

        tool.verify_warranty_status channel=commentary to=assistant:
        {"vin":"1HGCV1F32PA123456","status":"active","coverages":[{"type":"bumper_to_bumper","miles_remaining":17800,"months_remaining":12},{"type":"powertrain","miles_remaining":41800,"months_remaining":36}],"notes":"Brake components covered under bumper-to-bumper."}

        assistant channel=commentary to=tool.get_service_appointments:
        {"location_id":"autocare_sfo_01","service_type":"warranty_repair","start_date":"2025-08-23","end_date":"2025-08-24","preferred_time_of_day":"morning","loaner_required":true}

        tool.get_service_appointments channel=commentary to=assistant:
        {"slots":[{"slot_id":"slot_2025-08-23_08:00","start":"2025-08-23T08:00:00","end":"2025-08-23T12:00:00"},{"slot_id":"slot_2025-08-23_09:30","start":"2025-08-23T09:30:00","end":"2025-08-23T13:30:00"},{"slot_id":"slot_2025-08-23_11:00","start":"2025-08-23T11:00:00","end":"2025-08-23T15:00:00"}],"location_timezone":"America/Los_Angeles"}

        assistant channel=analysis:
        Pick the earliest suitable slot (08:00). Create the ticket as a warranty claim. Request a loaner for 6 hours. Then send SMS confirmation.

        assistant channel=commentary to=tool.create_service_ticket:
        {"customer_id":"cust_28917","vin":"1HGCV1F32PA123456","concern_description":"Grinding noise during braking; inspect pads/rotors/calipers under warranty.","warranty_claim":true,"appointment_slot_id":"slot_2025-08-23_08:00","contact_phone_e164":"+15551234567"}

        tool.create_service_ticket channel=commentary to=assistant:
        {"ticket_id":"TCK-AR-10492","status":"created","appointment_slot_id":"slot_2025-08-23_08:00"}

        assistant channel=commentary to=tool.request_loaner_vehicle:
        {"location_id":"autocare_sfo_01","date":"2025-08-23","duration_hours":6,"customer_id":"cust_28917","vin":"1HGCV1F32PA123456"}

        tool.request_loaner_vehicle channel=commentary to=assistant:
        {"reservation_id":"LNR-5521","status":"reserved","pickup_time":"2025-08-23T07:45:00"}

        assistant channel=commentary to=tool.send_confirmation_sms:
        {"phone_e164":"+15551234567","message":"AutoCare: Your warranty brake inspection is booked for Sat 08/23 8:00 AM (slot_2025-08-23_08:00). Ticket TCK-AR-10492. Loaner reserved (LNR-5521). Reply STOP to opt out.","ticket_id":"TCK-AR-10492"}

        tool.send_confirmation_sms channel=commentary to=assistant:
        {"status":"sent","message_id":"sms_9f2a7b"}

        assistant channel=analysis:
        Warranty verified as active; earliest morning slot available at 08:00 with loaner reserved. All confirmations complete; proceed to summarize to the customer and offer any additional preparation options.

        assistant channel=final:
        Thanks—here’s what I’ve set up:

        - Warranty: Active. Brake components are covered under bumper‑to‑bumper.
        - Appointment: Sat 08/23 at 8:00 AM (held for you).
        - Ticket: TCK-AR-10492.
        - Loaner: Reserved (pickup 7:45 AM).
        - SMS: Confirmation text sent to +1 555-123-4567.

        What else would you like me to prepare—pickup/drop-off notes, or adding any other concerns to the work order?
        """  # noqa: E501
    ).strip()

    messages = messages_from_plaintext_of_gpt_oss(text)
    for m in messages:
        print(m.model_dump_json(indent=4))
