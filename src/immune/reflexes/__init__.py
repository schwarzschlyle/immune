from __future__ import annotations

from immune.config.spec import Spec
from immune.reflexes.findings import Cleaned, Finding
from immune.reflexes.inbound import (
    EncodedPayloadDecoder,
    GuardAddressedDetector,
    HiddenMarkupStripper,
    TemplateTokenDetector,
)
from immune.reflexes.outbound import (
    ExfiltrationLinkDetector,
    GroundedNumbers,
    LinkInspector,
    MarkupSanitizer,
    PromptCopyDetector,
)
from immune.reflexes.patterns import PatternSet
from immune.reflexes.privacy import PersonalDataScanner, Redactor, SecretScanner
from immune.reflexes.tools import CommandInspector, DestinationExtractor
from immune.reflexes.unicode import UnicodeReveal


class ReflexSuite:
    def __init__(self, spec: Spec) -> None:
        template_tokens = PatternSet(spec.reflexes.template_tokens)
        self.unicode = UnicodeReveal()
        self.hidden_markup = HiddenMarkupStripper()
        self.encoded = EncodedPayloadDecoder()
        self.template_tokens = TemplateTokenDetector(template_tokens)
        self.guard_addressed = GuardAddressedDetector(PatternSet(spec.reflexes.guard_addressed), template_tokens)
        self.secrets = SecretScanner(PatternSet(spec.reflexes.secrets))
        self.personal = PersonalDataScanner()
        self.redactor = Redactor(self.secrets, self.personal)
        self.links = LinkInspector()
        self.exfiltration = ExfiltrationLinkDetector(self.links)
        self.markup = MarkupSanitizer()
        self.prompt_copy = PromptCopyDetector()
        self.numbers = GroundedNumbers()
        self.destinations = DestinationExtractor()
        self.commands = CommandInspector(PatternSet(spec.reflexes.sql), PatternSet(spec.reflexes.paths))


__all__ = ["Cleaned", "Finding", "ReflexSuite"]
