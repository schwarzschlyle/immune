from typing import Literal

from openai import OpenAI
from pydantic import BaseModel

import immune

immune.init()


class Triage(BaseModel):
    priority: Literal["low", "medium", "high"]
    refund_requested: bool


client = OpenAI()
with immune.site("ticket-triage"):
    response = client.responses.parse(
        model="gpt-5.5",
        instructions="Triage the support ticket.",
        input="The checkout page is down for every customer and we are losing sales!",
        text_format=Triage,
    )

verdict = immune.verdict(response)
print(response.output_parsed)
print("independent read of priority:", verdict.echo.get("priority"))
print(verdict.explanation)
