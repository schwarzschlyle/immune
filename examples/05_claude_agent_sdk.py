import asyncio

from claude_agent_sdk import ClaudeAgentOptions, query

from immune.integrations.claude_agent import hooks


async def main() -> None:
    options = ClaudeAgentOptions(system_prompt="You triage GitHub issues.", hooks=hooks())
    async for message in query(prompt="Triage the newest issue in this repository.", options=options):
        print(message)


asyncio.run(main())
