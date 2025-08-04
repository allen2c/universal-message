# Universal Message

A universal message format for AI interactions, designed to be simple, flexible, and easy to use.

## Installation

Install the package using pip:

```bash
pip install universal-message
```

## Usage

Create a message from a string:

```python
import universal_message as um

message = um.Message.from_any("Hello, world!")

print(message.to_instructions())
# user:
# Hello, world!
```

Create a message from a dictionary:

```python
import universal_message as um

message = um.Message.from_any(
    {
        "role": "assistant",
        "content": "I am a helpful assistant.",
    }
)

print(message.to_instructions())
# assistant:
# I am a helpful assistant.
```

## Features

* **Universal Format**: A single message format for all AI interactions.
* **Flexible**: Supports text, data URLs, and lists of content.
* **Easy to Use**: Create messages from strings, dictionaries, or pydantic models.
* **OpenAI Compatible**: Easily convert to and from OpenAI's chat completion format.
