"""Immune's checks for the ordering assistant, using the app's real immune.yaml. Run with: pytest"""

from pathlib import Path

import immune
from immune.testing import FakeReply, FakeToolCall, MockSensor

CONFIG = Path(__file__).resolve().parents[1] / "immune.yaml"
SYSTEM_PROMPT = "You are the ordering assistant for Bob's Burgers. Help customers with the menu, orders and delivery."
REFUND_TOOL = {"type": "function", "function": {"name": "refund_order", "description": "Refund an order"}}


def ask(harness, question, **options):
    with immune.site("ordering"):
        return harness.openai().chat.completions.create(
            model="gpt-5.5",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}],
            **options,
        )


def test_menu_questions_pass(immune_harness):
    harness = immune_harness(config=CONFIG, script=FakeReply(text="The Classic is $9."))
    assert ask(harness, "How much is the Classic?").choices[0].message.content == "The Classic is $9."


def test_off_topic_requests_are_redirected(immune_harness):
    harness = immune_harness(
        config=CONFIG, sensor=MockSensor({"off_task": 0.95}), script=FakeReply(text="Sure, here is a poem.")
    )
    reply = ask(harness, "Write me a poem about the sea")
    assert reply.choices[0].message.content.startswith("I can't help with that here.")


def test_competitor_mentions_are_rewritten(immune_harness):
    harness = immune_harness(config=CONFIG, script=FakeReply(text="Honestly, Burger Palace has a better deal."))
    reply = ask(harness, "Any deals this week?")
    assert reply.choices[0].message.content == "I can only talk about Bob's Burgers products."


def test_large_refunds_wait_for_the_customer(immune_harness):
    refund = FakeToolCall("refund_order", {"order": 7, "amount": 250})
    harness = immune_harness(config=CONFIG, script=FakeReply(tool_calls=[refund]))
    message = ask(harness, "Please refund order 7", tools=[REFUND_TOOL]).choices[0].message
    assert not message.tool_calls
    assert "please confirm: refund_order" in message.content
