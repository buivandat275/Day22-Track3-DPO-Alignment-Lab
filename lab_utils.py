"""Shared helpers for the Day 22 lab."""
from __future__ import annotations


DEFAULT_CHAT_TEMPLATE = """{% for message in messages %}{% if message['role'] == 'system' %}<|im_start|>system
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'user' %}<|im_start|>user
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'assistant' %}<|im_start|>assistant
{{ message['content'] }}<|im_end|>
{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"""


def ensure_chat_template(tokenizer):
    """Attach a simple ChatML-style template if the tokenizer doesn't ship one."""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer
    tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
    return tokenizer
