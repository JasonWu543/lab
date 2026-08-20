"""minisft/chat.py — ChatML rendering and SFT example builder.

学生任务：实现 render_chat 和 build_example。
"""
from __future__ import annotations

from minisft.tokenizer import ByteTokenizer, IM_START, IM_END, IM_START_ID, IM_END_ID

# ChatML 模板：每条消息的格式
_TEMPLATE = "<|im_start|>{role}\n{content}<|im_end|>\n"


def render_chat(messages: list[dict]) -> str:
    """
    将消息列表渲染成 ChatML 字符串。

    每条消息格式：
        <|im_start|>{role}\n{content}<|im_end|>\n

    示例：
        render_chat([{"role": "user", "content": "Hi"}])
        # => "<|im_start|>user\nHi<|im_end|>\n"

    Args:
        messages: list of {"role": str, "content": str}

    Returns:
        完整的 ChatML 字符串（各消息直接拼接）
    """
    # TODO: 遍历 messages，用 _TEMPLATE 格式化后拼接
    raise NotImplementedError("render_chat: 你来实现")


def build_example(
    tok: ByteTokenizer, messages: list[dict]
) -> tuple[list[int], list[int]]:
    """
    将对话 tokenize，返回 (input_ids, labels)。

    labels 规则：
    - labels 与 input_ids 等长
    - labels[i] 是 input_ids[i] 位置的"预测目标"
    - 只有 assistant 消息的内容 token 和紧跟其后的 <|im_end|> token 才有目标
    - 其余位置（user、system 头部、换行等）一律为 -100（不参与 loss）
    - labels[-1] 永远是 -100（最后一个位置没有下一个 token 可以预测）

    提示（不要跳过）：
    ① labels[i] 是 input_ids[i] 位置的预测目标——想清楚 assistant 段的哪些位置
      有目标、<|im_start|>assistant\n 这段头部算不算。
    ② 一个有用的构造思路：先把 labels 全部设为 -100，再把需要激活的位置改为
      input_ids[i+1]（下一个 token）。
    ③ 关键问题：如何知道 "assistant 内容" 从第几个 token 开始、到第几个结束？
      可以把每条消息单独 tokenize，用长度来定位游标。

    Args:
        tok: ByteTokenizer 实例
        messages: list of {"role": str, "content": str}

    Returns:
        (input_ids, labels)，两者等长，labels 元素为 int。
    """
    # TODO: 实现
    # 建议步骤：
    # 1. 用 render_chat(messages) 得到完整文本，tokenize -> input_ids
    # 2. 初始化 labels = [-100] * len(input_ids)
    # 3. 遍历 messages，用 tok.encode() 单独 tokenize 每条消息来确定 token 跨度
    # 4. 对 assistant 消息：找到 "<|im_start|>assistant\n" 前缀的长度
    #    以确定"内容"从哪个 token 开始
    # 5. 把 [content_start-1 .. im_end_pos-1] 范围内的 labels[i] = input_ids[i+1]
    # 6. 确保 labels[-1] = -100
    raise NotImplementedError("build_example: 你来实现")
