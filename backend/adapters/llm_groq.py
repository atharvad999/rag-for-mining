from typing import Optional


class GroqLLM:
    """Groq LLM adapter using groq SDK (chat.completions)."""

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def complete(self, prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
        try:
            from groq import Groq

            client = Groq(api_key=self.api_key)
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a tender assistant. Only answer from provided context."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception:
            return "Not found in tender"
