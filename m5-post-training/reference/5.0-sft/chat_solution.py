"""Reference solution for minisft/chat.py — do not read until stuck for 30+ min."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from minisft.tokenizer import ByteTokenizer, IM_START, IM_END, IM_START_ID, IM_END_ID


def render_chat(messages: list[dict]) -> str:
    """Render a list of messages into the ChatML string format."""
    result = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        result += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    return result


def build_example(tok: ByteTokenizer, messages: list[dict]) -> tuple[list[int], list[int]]:
    """
    Tokenize a conversation, returning (input_ids, labels).
    labels[i] = input_ids[i+1] for positions in assistant content+<|im_end|>, else -100.
    labels[-1] = -100 always.
    """
    full_text = render_chat(messages)
    input_ids = tok.encode(full_text)
    n = len(input_ids)

    # Start with all -100
    labels = [-100] * n

    # Walk through messages, tracking byte offset -> token position
    # We tokenize each message prefix to find exact token spans
    cursor = 0  # token cursor in input_ids

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        msg_text = f"<|im_start|>{role}\n{content}<|im_end|>\n"
        msg_ids = tok.encode(msg_text)

        if role == "assistant":
            # Find where the actual content starts (after "<|im_start|>assistant\n")
            prefix_text = f"<|im_start|>assistant\n"
            prefix_ids = tok.encode(prefix_text)
            prefix_len = len(prefix_ids)

            # content + <|im_end|> + \n  -> last two tokens of msg_ids are IM_END_ID and ord('\n')
            # We want: from (cursor + prefix_len - 1) through (cursor + len(msg_ids) - 2)
            # i.e., positions where the *next* token is the first content token through IM_END_ID
            # labels[i] = input_ids[i+1] for those i

            # Target range in input_ids:
            # content starts at: cursor + prefix_len
            # IM_END is at: cursor + len(msg_ids) - 2  (second to last, since last is '\n')
            # The '\n' after IM_END is NOT a target

            content_start = cursor + prefix_len
            # IM_END token position:
            im_end_pos = cursor + len(msg_ids) - 2  # msg ends with <|im_end|>\n

            # labels[i] = input_ids[i+1] for i in [content_start-1 .. im_end_pos-1]
            # because labels[i] is prediction target AT position i,
            # meaning: what should be predicted when we see input_ids[i]?
            # That's input_ids[i+1].
            # So we want labels[i] set for positions i where i+1 is in [content_start, im_end_pos]
            for i in range(content_start - 1, im_end_pos):
                if i + 1 < n:
                    labels[i] = input_ids[i + 1]

        cursor += len(msg_ids)

    # Ensure last position is -100
    labels[-1] = -100

    return input_ids, labels
