from __future__ import annotations

from dataclasses import dataclass

from immune.core.conversation import Conversation
from immune.profiling.profile import SiteProfile
from immune.profiling.registry import Site
from immune.reflexes.privacy import SecretScanner

_ANONYMOUS_SHARE = 0.5
_ANONYMOUS_MINIMUM_CALLS = 20


@dataclass(frozen=True, slots=True)
class PostureIssue:
    code: str
    message: str


class PostureAssessor:
    def __init__(self, secrets: SecretScanner) -> None:
        self._secrets = secrets

    def assess_profile(self, profile: SiteProfile, reads_untrusted: bool = False) -> tuple[PostureIssue, ...]:
        issues: list[PostureIssue] = []
        untrusted = reads_untrusted or "processes_external_content" in profile.attributes
        if untrusted and profile.any_capability("reads_private") and profile.any_capability("egress"):
            issues.append(
                PostureIssue(
                    "rule_of_two",
                    "reads untrusted content, reads private data and can send data out in one session",
                )
            )
        issues.extend(PostureIssue(f"allergy:{name}", name.replace("_", " ")) for name in profile.allergies)
        return tuple(issues)

    def assess_site(self, site: Site) -> tuple[PostureIssue, ...]:
        issues = list(self.assess_profile(site.profile))
        if (
            site.profile.user_facing
            and site.calls >= _ANONYMOUS_MINIMUM_CALLS
            and site.anonymous_share > _ANONYMOUS_SHARE
        ):
            issues.append(
                PostureIssue(
                    "anonymous_sessions",
                    f"{site.anonymous_share:.0%} of calls carry no session identity; "
                    "use immune.session() so multi-turn defenses apply",
                )
            )
        return tuple(issues)

    def assess(self, profile: SiteProfile, conversation: Conversation) -> tuple[PostureIssue, ...]:
        issues = list(self.assess_profile(profile, conversation.has_context))
        if any(True for _ in self._secrets.find(conversation.operator_text)):
            issues.append(PostureIssue("secret_in_prompt", "the operator prompt contains a credential"))
        if any(segment.is_embedded and segment.confidence < 1.0 for segment in conversation.data):
            issues.append(PostureIssue("unmarked_data", "pasted data found without immune.untrusted() markers"))
        return tuple(issues)
