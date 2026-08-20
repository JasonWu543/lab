"""ByteTokenizer —— M5 全模块共用的给定脚手架（不是学生任务）。

字节级 tokenizer + 3 个原子特殊 token：
    <|im_start|> = 256    <|im_end|> = 257    <|pad|> = 258（5.2 里兼作 EOS）
vocab_size = 259。特殊 token 在 encode 时原子切分、decode 时原样还原。
"""
from __future__ import annotations

import re

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
PAD = "<|pad|>"

IM_START_ID = 256
IM_END_ID = 257
PAD_ID = 258

VOCAB_SIZE = 259

_SPECIAL_TO_ID = {IM_START: IM_START_ID, IM_END: IM_END_ID, PAD: PAD_ID}
_ID_TO_SPECIAL = {v: k for k, v in _SPECIAL_TO_ID.items()}
_SPLIT_RE = re.compile(r"(<\|im_start\|>|<\|im_end\|>|<\|pad\|>)")


class ByteTokenizer:
    vocab_size = VOCAB_SIZE
    pad_id = PAD_ID
    eos_id = PAD_ID  # 5.2 里 <|pad|> 兼作 EOS

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in _SPLIT_RE.split(text):
            if not piece:
                continue
            if piece in _SPECIAL_TO_ID:
                ids.append(_SPECIAL_TO_ID[piece])
            else:
                ids.extend(piece.encode("utf-8"))
        return ids

    def decode(self, ids: list[int]) -> str:
        out: list[str] = []
        buf = bytearray()
        for i in ids:
            if i in _ID_TO_SPECIAL:
                if buf:
                    out.append(buf.decode("utf-8", errors="replace"))
                    buf = bytearray()
                out.append(_ID_TO_SPECIAL[i])
            else:
                buf.append(i)
        if buf:
            out.append(buf.decode("utf-8", errors="replace"))
        return "".join(out)
