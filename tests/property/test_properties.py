from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from immune.config.spec import FeatureSpec, HeadSpec
from immune.core.sse import ServerSentEvent, ServerSentEvents
from immune.heads import Head
from immune.reflexes import ReflexSuite
from immune.sensing.signals import SensorReading, Signal
from immune.types import Action

probabilities = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
text = st.text(max_size=400)


@given(text)
@settings(max_examples=300)
def test_unicode_reveal_is_idempotent_and_total(reflexes: ReflexSuite, value: str) -> None:
    once = reflexes.unicode.reveal(value).text
    assert reflexes.unicode.reveal(once).text == once


@given(text)
@settings(max_examples=200)
def test_hidden_markup_stripping_is_idempotent(reflexes: ReflexSuite, value: str) -> None:
    once = reflexes.hidden_markup.clean(value).text
    assert reflexes.hidden_markup.clean(once).text == once


@given(text)
@settings(max_examples=200)
def test_outbound_reflexes_never_raise(reflexes: ReflexSuite, value: str) -> None:
    list(reflexes.exfiltration.findings(value, ["example.test"]))
    list(reflexes.markup.findings(value))
    list(reflexes.secrets.find(value))
    reflexes.redactor.mask(value)
    reflexes.guard_addressed.defang(value)


@given(
    st.lists(
        st.tuples(
            st.text(alphabet=st.characters(blacklist_characters="\r\n"), max_size=40),
            st.one_of(st.none(), st.sampled_from(["delta", "message_start"])),
        ),
        max_size=10,
    )
)
def test_server_sent_events_round_trip(events: list[tuple[str, str | None]]) -> None:
    original = [ServerSentEvent(data=data or "x", event=name) for data, name in events]
    assert ServerSentEvents.parse(ServerSentEvents.serialize(original)) == original


@given(st.lists(st.sampled_from(list(Action)), min_size=1))
def test_most_severe_action_dominates(actions: list[Action]) -> None:
    chosen = Action.most_severe(actions)
    assert all(chosen.severity >= action.severity for action in actions)


@given(probabilities, probabilities)
def test_heads_are_monotone_in_their_signal(low: float, high: float) -> None:
    low, high = sorted((low, high))
    head = Head("t", HeadSpec(features=(FeatureSpec(signal="x"), FeatureSpec(signal="y", weight=0.5))))

    def score(value: float) -> float:
        reading = SensorReading({"x": Signal("x", "noul", value), "y": Signal("y", "noul", 0.3)}, "test")
        result = head.score(reading, set())
        assert result is not None
        return result.probability

    assert score(low) <= score(high) + 1e-12
