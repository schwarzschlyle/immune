from immune.codecs.anthropic_messages import AnthropicMessagesCodec
from immune.codecs.base import Codec, ReplyEdit, RequestContext
from immune.codecs.bedrock_converse import BedrockConverseCodec
from immune.codecs.gemini import GeminiCodec
from immune.codecs.openai_chat import OpenAIChatCodec
from immune.codecs.openai_responses import OpenAIResponsesCodec

CODECS: dict[str, Codec] = {
    codec.name: codec
    for codec in (
        OpenAIChatCodec(),
        OpenAIResponsesCodec(),
        AnthropicMessagesCodec(),
        GeminiCodec(),
        BedrockConverseCodec(),
    )
}

__all__ = [
    "CODECS",
    "AnthropicMessagesCodec",
    "BedrockConverseCodec",
    "Codec",
    "GeminiCodec",
    "OpenAIChatCodec",
    "OpenAIResponsesCodec",
    "ReplyEdit",
    "RequestContext",
]
