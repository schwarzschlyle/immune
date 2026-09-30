from openai import OpenAI

import immune

immune.init()

client = OpenAI()
reply = client.chat.completions.create(
    model="gpt-5.5",
    messages=[
        {"role": "system", "content": "You are the support assistant for Acme Burgers."},
        {"role": "user", "content": "What vegetarian options do you have?"},
    ],
)

verdict = immune.verdict(reply)
print(reply.choices[0].message.content)
print(verdict.action.value, "-", verdict.explanation)
