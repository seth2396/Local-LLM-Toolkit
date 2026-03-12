from typing import List, Dict
import logging
from openai import OpenAI

from .BaseAgent import BaseAgent
from .tools import BaseTool


class ChatAgent(BaseAgent):
    def __init__(self, system_prompt: str, client: OpenAI, model: str, tools: BaseTool | list[BaseTool] = None, temperature: float = 0.2, stream: bool = False):
        super().__init__(system_prompt=system_prompt, client=client, model=model, tools=tools, temperature=temperature, stream=stream)
        self.history = []

    def chat(self, message: str, history: List[Dict[str, str]] = None):
        if history:
            self.history = history

        if not self.stream:
            response_text = super().chat(message=message, history=self.history)

            self.history += [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response_text}
            ]
            return response_text

        else:
            response_gen = super().chat(message=message, history=self.history)

            def generator():
                response_tokens = []
                for token in response_gen:
                    if token:
                        response_tokens.append(token)
                        yield token

                response_text = "".join(response_tokens)
                self.history += [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": response_text},
                ]
            return generator()

    def inject(self, message, inject):
        response_text = super().inject(message, inject)
        self.history += [{"role": "user", "content": message.strip()}, {"role": "assistant", "content": response_text}]
        return response_text

    def clear_history(self):
        logging.info("Chat history reset")
        self.history = [{"role": "system", "content": self.system_prompt}]
