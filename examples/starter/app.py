"""The ordering assistant for Bob's Burgers, protected by Immune."""

from openai import OpenAI

import immune

immune.init()  # reads ./immune.yaml (or IMMUNE_CONFIG) and loads vaccines/

SYSTEM_PROMPT = "You are the ordering assistant for Bob's Burgers. Help customers with the menu, orders and delivery."
client = OpenAI()


def answer(question: str, user_id: str) -> str:
    with immune.site("ordering"), immune.session(user_id):
        completion = client.chat.completions.create(
            model="gpt-5.5",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}],
        )
    return completion.choices[0].message.content or ""


if __name__ == "__main__":
    print(answer("What's on the menu?", user_id="demo"))
