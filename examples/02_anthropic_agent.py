from anthropic import Anthropic

import immune

immune.init()

TOOLS = [
    {"name": "read_inbox", "description": "Read the user's unread emails", "input_schema": {"type": "object"}},
    {
        "name": "send_email",
        "description": "Send an email",
        "input_schema": {"type": "object", "properties": {"to": {"type": "string"}, "body": {"type": "string"}}},
    },
]

client = Anthropic()
with immune.session("user-42"), immune.site("inbox-agent"):
    message = client.messages.create(
        model="claude-opus-5",
        max_tokens=16000,
        system="You are an email assistant. Use tools when needed.",
        tools=TOOLS,
        messages=[{"role": "user", "content": "Summarize my inbox and send the summary to boss@acme.test"}],
    )

for block in message.content:
    print(block.type, getattr(block, "text", getattr(block, "input", "")))
print(immune.verdict(message).explanation)
