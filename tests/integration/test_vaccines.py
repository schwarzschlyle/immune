from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx2
import openai
import pytest
import yaml

import immune
from immune.errors import ConfigError
from immune.testing import FakeProvider, FakeReply, FakeToolCall, ImmuneHarness, MockSensor
from immune.types import Action
from tests.conftest import OPERATOR

Factory = Callable[..., ImmuneHarness]
COMPETITOR: dict[str, Any] = {
    "id": "acme.no_competitor_mentions",
    "title": "The reply names a competitor",
    "stage": "output",
    "detect": {"keywords": ["Burger Palace"]},
    "respond": {"message": "I can only talk about Acme Burgers products."},
}


def vaccine(directory: Path, document: dict[str, Any]) -> Path:
    path = directory / f"{document['id']}.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def chat(harness: ImmuneHarness, user: str, **options: Any) -> Any:
    return harness.openai().chat.completions.create(
        model="gpt-5.5",
        messages=[{"role": "system", "content": OPERATOR}, {"role": "user", "content": user}],
        **options,
    )


def agent(harness: ImmuneHarness, tools: list[str], user: str = "Please handle my account.") -> Any:
    specs = [{"type": "function", "function": {"name": name, "description": name.replace("_", " ")}} for name in tools]
    return chat(harness, user, tools=specs)


def threats(harness: ImmuneHarness) -> dict[str, bool]:
    verdict = harness.verdict()
    assert verdict is not None
    return {hit.threat: hit.enforced for hit in verdict.hits}


class TestDetectors:
    def test_keyword_vaccines_start_observed(self, immune_harness: Factory, tmp_path: Path) -> None:
        path = vaccine(tmp_path, COMPETITOR)
        harness = immune_harness(
            script=FakeReply(text="You might prefer Burger Palace."), config={"vaccines": {"paths": [str(path)]}}
        )
        reply = chat(harness, "Any recommendations?")
        assert reply.choices[0].message.content == "You might prefer Burger Palace."
        assert threats(harness) == {"acme.no_competitor_mentions": False}

    def test_enforced_vaccines_use_their_own_message(self, immune_harness: Factory, tmp_path: Path) -> None:
        path = vaccine(tmp_path, {**COMPETITOR, "enforcement": "enforce"})
        harness = immune_harness(
            script=FakeReply(text="You might prefer Burger Palace."), config={"vaccines": {"paths": [str(path)]}}
        )
        reply = chat(harness, "Any recommendations?")
        assert reply.choices[0].message.content == "I can only talk about Acme Burgers products."
        verdict = harness.verdict()
        assert verdict is not None
        assert verdict.action is Action.REWRITE
        assert "keyword 'Burger Palace'" in verdict.hits[0].evidence

    def test_confirmed_vaccines_fire_only_when_jev_agrees(self, immune_harness: Factory, tmp_path: Path) -> None:
        document = {**COMPETITOR, "enforcement": "enforce", "detect": {"keywords": ["Burger Palace"], "confirm": "jev"}}
        path = vaccine(tmp_path, document)
        config = {"vaccines": {"paths": [str(path)]}}
        text = "We're across the road from Burger Palace."
        declined = immune_harness(
            sensor=MockSensor({"acme_no_competitor_mentions__confirmed": 0.1}),
            script=FakeReply(text=text),
            config=config,
        )
        assert chat(declined, "Where are you?").choices[0].message.content == text
        assert "acme.no_competitor_mentions" not in threats(declined)
        sensor = MockSensor({"acme_no_competitor_mentions__confirmed": 0.95})
        agreed = immune_harness(sensor=sensor, script=FakeReply(text=text), config=config)
        assert chat(agreed, "Where are you?").choices[0].message.content == COMPETITOR["respond"]["message"]
        [(state, keys)] = [(state, keys) for state, keys in sensor.calls if any(k.startswith("cand_") for k in keys)]
        assert keys == ("cand_1__acme_no_competitor_mentions__confirmed",)
        assert state["cand_1"]["check"] == "acme.no_competitor_mentions"

    def test_confirm_only_applies_to_rules(self, tmp_path: Path) -> None:
        document = {
            **COMPETITOR,
            "detect": {"questions": [{"key": "rival", "text": "It names a rival."}], "confirm": "jev"},
        }
        with pytest.raises(ConfigError, match="confirm: jev"):
            ImmuneHarness(tmp_path / "state", config={"vaccines": {"paths": [str(vaccine(tmp_path, document))]}})

    def test_input_vaccines_redirect_before_the_model_replies(self, immune_harness: Factory, tmp_path: Path) -> None:
        document = {
            "id": "acme.no_account_numbers",
            "title": "Account numbers pasted into chat",
            "stage": "input",
            "detect": {"regex": [r"\bACC-\d{6}\b"]},
            "enforcement": "enforce",
            "respond": {"message": "Please don't share account numbers here."},
        }
        harness = immune_harness(config={"vaccines": {"paths": [str(vaccine(tmp_path, document))]}})
        reply = chat(harness, "My account is ACC-123456")
        assert reply.choices[0].message.content == "Please don't share account numbers here."

    def test_data_vaccines_neutralize_the_item(self, immune_harness: Factory, tmp_path: Path) -> None:
        document = {
            "id": "acme.internal_markers",
            "title": "Internal-only documents",
            "stage": "data",
            "detect": {"keywords": ["INTERNAL ONLY"]},
            "enforcement": "enforce",
        }
        harness = immune_harness(config={"vaccines": {"paths": [str(vaccine(tmp_path, document))]}})
        harness.openai().chat.completions.create(
            model="gpt-5.5",
            messages=[
                {"role": "system", "content": OPERATOR},
                {"role": "user", "content": "Summarize the policy"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "t0", "type": "function", "function": {"name": "read_doc", "arguments": "{}"}}
                    ],
                },
                {"role": "tool", "tool_call_id": "t0", "content": "INTERNAL ONLY: margins by store"},
            ],
        )
        forwarded = harness.provider.last_request["messages"][3]["content"]
        assert forwarded.startswith("[Immune removed content from")

    def test_question_vaccines_join_the_jev_panel(self, immune_harness: Factory, tmp_path: Path) -> None:
        document = {
            "id": "acme.no_dosage_advice",
            "title": "Dosage advice",
            "stage": "output",
            "detect": {
                "questions": [{"key": "gives_dosage", "text": "The assistant output tells the user how much to take."}],
                "threshold": 0.8,
            },
            "applies_to": {"sites": ["pharmacy"]},
            "enforcement": "enforce",
        }
        sensor = MockSensor({"acme_no_dosage_advice__gives_dosage": 0.95})
        harness = immune_harness(
            sensor=sensor,
            script=FakeReply(text="Take two tablets."),
            config={"vaccines": {"paths": [str(vaccine(tmp_path, document))]}},
        )
        with immune.site("pharmacy"):
            reply = chat(harness, "How much should I take?")
        assert reply.choices[0].message.content != "Take two tablets."
        assert threats(harness) == {"acme.no_dosage_advice": True}
        asked = [keys for _, keys in sensor.calls if "acme_no_dosage_advice__gives_dosage" in keys]
        assert len(asked) == 1
        with immune.site("kiosk"):
            chat(harness, "How much should I take?")
        assert "acme.no_dosage_advice" not in threats(harness)

    def test_tool_rules_ask_for_confirmation(self, immune_harness: Factory, tmp_path: Path) -> None:
        document = {
            "id": "acme.refund_over_limit",
            "title": "Refunds over $100 need confirmation",
            "stage": "tool",
            "detect": {"tool": "refund_order", "argument": {"path": "amount", "greater_than": 100}},
            "respond": {"tool": "confirm"},
            "enforcement": "enforce",
        }
        path = vaccine(tmp_path, document)
        large = immune_harness(
            sensor=MockSensor({"talks_to_end_users": 0.95}),
            script=FakeReply(tool_calls=[FakeToolCall("refund_order", {"order": 7, "amount": 250})]),
            config={"vaccines": {"paths": [str(path)]}},
        )
        message = agent(large, ["refund_order"]).choices[0].message
        assert not message.tool_calls
        assert "please confirm: refund_order(amount=250, order=7)" in (message.content or "")
        small = immune_harness(
            script=FakeReply(tool_calls=[FakeToolCall("refund_order", {"order": 7, "amount": 50})]),
            config={"vaccines": {"paths": [str(path)]}},
        )
        assert agent(small, ["refund_order"]).choices[0].message.tool_calls

    def test_python_vaccines_hold_calls_with_their_message(self, immune_harness: Factory, tmp_path: Path) -> None:
        document = {
            "id": "acme.vip_accounts",
            "title": "VIP account changes",
            "stage": "tool",
            "detect": {"python": "tests.integration.vaccine_helpers:touches_vip_account"},
            "respond": {"tool": "hold", "message": "VIP accounts are handled by the account team"},
            "enforcement": "enforce",
        }
        harness = immune_harness(
            script=FakeReply(tool_calls=[FakeToolCall("update_account", {"account": "vip-42"})]),
            config={"vaccines": {"paths": [str(vaccine(tmp_path, document))]}},
        )
        message = agent(harness, ["update_account"]).choices[0].message
        assert not message.tool_calls
        assert "VIP accounts are handled by the account team" in (message.content or "")
        verdict = harness.verdict()
        assert verdict is not None
        assert verdict.hits[0].evidence == ("account vip-42 is a VIP account",)

    def test_a_failing_python_vaccine_is_skipped(
        self, immune_harness: Factory, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        document = {
            "id": "acme.broken",
            "title": "Broken",
            "stage": "tool",
            "detect": {"python": "tests.integration.vaccine_helpers:always_fails"},
            "enforcement": "enforce",
        }
        harness = immune_harness(
            script=FakeReply(tool_calls=[FakeToolCall("update_account", {"account": "a-1"})]),
            config={"vaccines": {"paths": [str(vaccine(tmp_path, document))]}},
        )
        assert agent(harness, ["update_account"]).choices[0].message.tool_calls
        assert "vaccine acme.broken raised RuntimeError" in caplog.text


class TestSwitches:
    def test_built_in_threats_can_be_switched_off(self, immune_harness: Factory) -> None:
        harness = immune_harness(
            sensor=MockSensor({"off_task": 0.95}), mode="strict", config={"vaccines": {"disabled": ["input.off_task"]}}
        )
        chat(harness, "Write me a poem")
        assert "input.off_task" not in threats(harness)

    def test_switching_off_the_floor_needs_permission(self, immune_harness: Factory) -> None:
        with pytest.raises(ConfigError, match="floor protections"):
            immune_harness(config={"vaccines": {"disabled": ["output.secret_leak"]}})
        harness = immune_harness(
            script=FakeReply(text="Use AKIAABCDEFGHIJKLMNOP"),
            config={"vaccines": {"disabled": ["output.secret_leak"], "allow_floor_changes": True}},
        )
        assert chat(harness, "Key?").choices[0].message.content == "Use AKIAABCDEFGHIJKLMNOP"

    def test_default_off_vaccines_need_enabling(self, immune_harness: Factory, tmp_path: Path) -> None:
        path = vaccine(tmp_path, {**COMPETITOR, "default": "off"})
        script = FakeReply(text="Try Burger Palace.")
        off = immune_harness(script=script, config={"vaccines": {"paths": [str(path)]}})
        chat(off, "Ideas?")
        assert threats(off) == {}
        on = immune_harness(
            script=script,
            config={"vaccines": {"paths": [str(path)]}, "sites": {"menu": {"vaccines": {"enabled": ["acme.*"]}}}},
        )
        with immune.site("menu"):
            chat(on, "Ideas?")
        assert threats(on) == {"acme.no_competitor_mentions": False}

    def test_switches_change_live(self, tmp_path: Path) -> None:
        path = vaccine(tmp_path, {**COMPETITOR, "enforcement": "enforce"})
        immune.init(sensor=MockSensor(), state_dir=tmp_path / "state", vaccines=[path])
        try:
            runtime = immune.runtime()
            assert runtime is not None
            assert "vaccines-" in runtime.spec.digest
            provider = FakeProvider(FakeReply(text="Try Burger Palace."))
            client = openai.OpenAI(
                api_key="sk-test", http_client=httpx2.Client(transport=provider.transport(httpx2)), max_retries=0
            )

            def reply() -> str:
                messages = [{"role": "system", "content": OPERATOR}, {"role": "user", "content": "Ideas?"}]
                return client.chat.completions.create(model="m", messages=messages).choices[0].message.content or ""

            assert reply() == "I can only talk about Acme Burgers products."
            immune.configure(vaccines={"disabled": ["acme.no_competitor_mentions"]})
            assert reply() == "Try Burger Palace."
            with pytest.raises(ConfigError, match=r"vaccines.paths cannot change at runtime"):
                immune.configure(vaccines={"paths": []})
        finally:
            immune.shutdown()


class TestPinnedSitesAcrossRestarts:
    def test_a_pinned_user_facing_site_keeps_asking_after_a_restart(self, tmp_path: Path) -> None:
        document = {
            "id": "acme.refund_over_limit",
            "title": "Large refunds",
            "stage": "tool",
            "detect": {"tool": "refund_order", "argument": {"path": "amount", "greater_than": 100}},
            "respond": {"tool": "confirm"},
            "enforcement": "enforce",
        }
        config = {
            "vaccines": {"paths": [str(vaccine(tmp_path, document))]},
            "sites": {"shop": {"user_facing": True, "archetype": "customer_service"}},
        }
        state = tmp_path / "state"
        replies = []
        for _ in range(2):
            script = FakeReply(tool_calls=[FakeToolCall("refund_order", {"order": 7, "amount": 250})])
            with ImmuneHarness(state, script=script, config=config) as harness, immune.site("shop"):
                message = agent(harness, ["refund_order"]).choices[0].message
            replies.append(message.content or "")
        assert all("please confirm: refund_order" in reply for reply in replies), replies
