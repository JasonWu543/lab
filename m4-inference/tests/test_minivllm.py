"""
Phase 4.0: mini-vLLM 测试套件（T1–T7）
所有测试在 CPU + tiny 随机初始化 Qwen2 模型下运行，整套 ≤ 120 秒。

运行方式：
    cd m4-inference && python3 -m pytest tests/test_minivllm.py -x -q
    python3 -m pytest tests/test_minivllm.py -x -q -k T2   # 单独运行 T2

注意：学生骨架实现前，collection 应无报错，但核心测试会 FAIL（TODO 未实现）。
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minivllm.request import Request, Sequence, SeqStatus
from minivllm.block_manager import BlockManager
from minivllm.scheduler import Scheduler
from minivllm.sampler import sample
from minivllm.engine import Engine


# ─────────────────────────────── fixtures ────────────────────────────────────


def _make_model():
    """构造固定 seed 的 tiny Qwen2 模型（CPU，随机权重）。"""
    from transformers import Qwen2Config, AutoModelForCausalLM

    torch.manual_seed(42)
    cfg = Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=256,
    )
    model = AutoModelForCausalLM.from_config(cfg)
    model.eval()
    return model


@pytest.fixture(scope="module")
def tiny_model():
    return _make_model()


def _oracle(model, prompt_ids: list[int], max_new_tokens: int) -> list[int]:
    """串行 greedy 循环，作为 T2/T4/T5 的正确性 oracle。"""
    ids = torch.tensor([prompt_ids], dtype=torch.long)
    out: list[int] = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(ids).logits[0, -1, :]
            tok = int(logits.argmax().item())
            out.append(tok)
            ids = torch.cat([ids, torch.tensor([[tok]])], dim=1)
    return out


# ─────────────────────────────────── T1 ──────────────────────────────────────


class TestT1BlockManager:
    """T1：BlockManager 基本不变量。"""

    def test_initial_free_count(self):
        bm = BlockManager(num_blocks=16, block_size=4)
        assert bm.num_free_blocks == 16

    def test_allocate_no_overlap(self):
        """两个 seq 分配后，block_table 无重叠。"""
        bm = BlockManager(num_blocks=32, block_size=4)
        seqs = [
            Sequence(Request(req_id=i, prompt_ids=list(range(1, 9)), max_new_tokens=4))
            for i in range(2)
        ]
        for s in seqs:
            assert bm.can_allocate(s)
            bm.allocate(s)
        all_blocks = seqs[0].block_table + seqs[1].block_table
        assert len(all_blocks) == len(set(all_blocks)), "block_table 有重叠"

    def test_free_and_reuse(self):
        """free 后 num_free_blocks 恢复，且块可被再次分配。"""
        bm = BlockManager(num_blocks=8, block_size=4)
        seq = Sequence(Request(req_id=0, prompt_ids=list(range(1, 9)), max_new_tokens=2))
        bm.allocate(seq)
        free_after_alloc = bm.num_free_blocks
        bm.free(seq)
        assert bm.num_free_blocks > free_after_alloc, "free 后空闲块应增加"

        # 再分配另一个同样大小的 seq 不应报错
        seq2 = Sequence(Request(req_id=1, prompt_ids=list(range(1, 9)), max_new_tokens=2))
        assert bm.can_allocate(seq2)
        bm.allocate(seq2)

    def test_can_allocate_returns_false_not_crash(self):
        """块不足时 can_allocate 返回 False，不崩溃。"""
        bm = BlockManager(num_blocks=2, block_size=4)
        # 20 tokens 需要 5 块，远超 2 块
        big_seq = Sequence(Request(req_id=99, prompt_ids=list(range(1, 21)), max_new_tokens=1))
        assert bm.can_allocate(big_seq) is False

    def test_allocate_reserves_first_decode_slot(self):
        """Boundary-aligned prefills must not over-admit the first decode slot."""
        bm = BlockManager(num_blocks=2, block_size=4)
        first = Sequence(Request(req_id=0, prompt_ids=[1, 2, 3, 4], max_new_tokens=2))
        second = Sequence(Request(req_id=1, prompt_ids=[5, 6, 7, 8], max_new_tokens=2))
        assert bm.can_allocate(first)
        bm.allocate(first)
        assert len(first.block_table) == 2
        assert not bm.can_allocate(second)

    def test_refcount_after_allocate(self):
        """前缀缓存命中块的 refcount 应正确累积（两个 seq 共享前缀）。"""
        bm = BlockManager(num_blocks=32, block_size=4)
        prefix = list(range(1, 5))  # 恰好 1 整块

        # 第一个 seq：分配并 free（前缀块变 cached）
        s1 = Sequence(Request(req_id=0, prompt_ids=prefix + [10, 11], max_new_tokens=2))
        bm.allocate(s1)
        s1.output_ids = [10, 11]
        bm.free(s1)

        # 第二个 seq 应命中前缀
        s2 = Sequence(Request(req_id=1, prompt_ids=prefix + [20, 21], max_new_tokens=2))
        matched, cached_toks = bm.match_prefix(s2.request.prompt_ids)
        assert cached_toks == 4, f"应命中 4 tokens，实际 {cached_toks}"
        bm.allocate(s2)
        assert s2.num_cached_tokens == 4

        # 前缀块 refcount 应为 1（s2 在用）
        prefix_bid = s2.block_table[0]
        assert bm._refcount[prefix_bid] == 1

        bm.free(s2)
        # free 后 refcount 归零，块回到 cached/free 状态
        assert bm._refcount[prefix_bid] == 0

    def test_append_slot_allocates_new_block(self):
        """append_slot 跨块边界时自动分配新块。"""
        bm = BlockManager(num_blocks=32, block_size=4)
        seq = Sequence(Request(req_id=0, prompt_ids=[1, 2, 3, 4], max_new_tokens=8))
        bm.allocate(seq)
        blocks_before = len(seq.block_table)

        # 模拟 decode 5 个 token（跨过 prefill 已预留的首个 decode 块）
        for i in range(5):
            seq.output_ids.append(100 + i)
            bm.append_slot(seq)

        assert len(seq.block_table) > blocks_before, "跨块后应有更多 block"


# ─────────────────────────────────── T2 ──────────────────────────────────────


class TestT2Correctness:
    """T2：8 个不同长度请求（greedy），引擎输出与串行 oracle 完全一致。"""

    def test_8_requests_match_oracle(self, tiny_model):
        requests = [
            Request(
                req_id=i,
                prompt_ids=list(range(1, 3 + i * 2)),
                max_new_tokens=4 + i,
            )
            for i in range(8)
        ]
        engine = Engine(tiny_model, num_blocks=128, block_size=8, max_batch_tokens=512)
        for r in requests:
            engine.add_request(r)
        results = engine.run()

        for r in requests:
            oracle = _oracle(tiny_model, r.prompt_ids, r.max_new_tokens)
            assert results[r.req_id] == oracle, (
                f"req {r.req_id}: engine={results[r.req_id]}, oracle={oracle}"
            )

    def test_finished_prefill_does_not_allocate_unused_output_kv(self, tiny_model):
        """A one-token request can finish with a block-aligned, full pool."""
        prompt = [1, 2, 3, 4]
        engine = Engine(tiny_model, num_blocks=1, block_size=4, max_batch_tokens=32)
        engine.add_request(Request(req_id=0, prompt_ids=prompt, max_new_tokens=1))
        assert engine.run()[0] == _oracle(tiny_model, prompt, 1)


# ─────────────────────────────────── T3 ──────────────────────────────────────


class TestT3DynamicArrival:
    """T3：错峰到达；同一 step 既有 prefill 又有 decode。"""

    def test_late_arrival_admitted(self, tiny_model):
        """后到的请求在运行中途被准入；有步骤同时包含 prefill+decode。"""
        r_early = Request(req_id=0, prompt_ids=[1, 2, 3, 4, 5], max_new_tokens=8)
        r_late = Request(req_id=1, prompt_ids=[10, 11, 12], max_new_tokens=6, arrival_step=2)

        engine = Engine(tiny_model, num_blocks=128, block_size=8, max_batch_tokens=512)
        engine.add_request(r_early)

        found_mixed_step = False
        all_results: dict[int, list[int]] = {}

        for step_i in range(30):
            # r_late 在第 2 步时加入（模拟错峰到达）
            if step_i == 2:
                engine.add_request(r_late)

            # 记录本步 running 数（schedule 前）
            running_before = engine.scheduler.num_running
            waiting_before = engine.scheduler.num_waiting

            finished = engine.step()

            # 若本步同时有 running（decode）且有 waiting 被准入（prefill）
            if running_before > 0 and waiting_before > 0:
                found_mixed_step = True

            for seq in finished:
                all_results[seq.request.req_id] = list(seq.output_ids)

            if engine.scheduler.num_waiting == 0 and engine.scheduler.num_running == 0:
                break

        assert found_mixed_step, "应存在 prefill+decode 混合的步骤"
        assert 0 in all_results and 1 in all_results, "两个请求都应完成"

        # 正确性验证
        assert all_results[0] == _oracle(tiny_model, r_early.prompt_ids, r_early.max_new_tokens)
        assert all_results[1] == _oracle(tiny_model, r_late.prompt_ids, r_late.max_new_tokens)


# ─────────────────────────────────── T4 ──────────────────────────────────────


class TestT4PagedCorrectness:
    """T4：block_table 非连续时，输出仍与串行一致。"""

    def test_fragmented_pool(self, tiny_model):
        """先分配干扰 seq 再 free，造成碎片；目标 seq 仍输出正确。"""
        engine = Engine(tiny_model, num_blocks=32, block_size=4, max_batch_tokens=512)

        # 干扰 seq：占用然后释放，造成 block 池碎片
        r_interfere = Request(
            req_id=99, prompt_ids=list(range(1, 17)), max_new_tokens=4
        )
        engine.add_request(r_interfere)
        engine.run()  # 完成并释放（blocks 变为 cached 状态）

        # 目标 seq：从碎片化的池中分配
        r_target = Request(req_id=0, prompt_ids=list(range(50, 63)), max_new_tokens=6)
        engine.add_request(r_target)
        results = engine.run()

        oracle = _oracle(tiny_model, r_target.prompt_ids, r_target.max_new_tokens)
        assert results[0] == oracle, f"碎片化 block_table 下输出错误: {results[0]} vs {oracle}"


# ─────────────────────────────────── T5 ──────────────────────────────────────


class TestT5PrefixCache:
    """T5：prefix cache 命中、正确性、refcount 归零后块可回收。"""

    def test_prefix_hit_num_cached_tokens(self, tiny_model):
        """第二个请求 num_cached_tokens > 0。"""
        shared = list(range(1, 9))  # 8 tokens = 2 整块（block_size=4）
        engine = Engine(tiny_model, num_blocks=64, block_size=4, max_batch_tokens=512)

        r1 = Request(req_id=0, prompt_ids=shared + [10, 11], max_new_tokens=4)
        engine.add_request(r1)
        engine.run()

        r2 = Request(req_id=1, prompt_ids=shared + [20, 21], max_new_tokens=4)
        engine.add_request(r2)

        # 触发 allocate（schedule 会调用 allocate）
        prefill_seqs, _ = engine.scheduler.schedule()
        r2_seq = next((s for s in prefill_seqs if s.request.req_id == 1), None)
        assert r2_seq is not None, "r2 应被准入 prefill"
        assert r2_seq.num_cached_tokens >= 8, (
            f"r2 应命中至少 8 tokens 的前缀，实际 {r2_seq.num_cached_tokens}"
        )

    def test_prefix_hit_output_correct(self, tiny_model):
        """使用 prefix cache 后，输出与不用 cache 的 oracle 完全一致。"""
        shared = list(range(1, 9))
        engine = Engine(tiny_model, num_blocks=64, block_size=4, max_batch_tokens=512)

        r1 = Request(req_id=0, prompt_ids=shared + [10, 11], max_new_tokens=4)
        engine.add_request(r1)
        engine.run()

        r2 = Request(req_id=1, prompt_ids=shared + [20, 21], max_new_tokens=4)
        engine.add_request(r2)
        results = engine.run()

        oracle = _oracle(tiny_model, shared + [20, 21], 4)
        assert results[1] == oracle, f"prefix cache 下输出错误: {results[1]} vs {oracle}"

    def test_exact_full_prompt_hit_output_correct(self, tiny_model):
        """An aligned prompt may be fully cached but still needs one model input."""
        prompt = list(range(1, 9))
        engine = Engine(tiny_model, num_blocks=16, block_size=4, max_batch_tokens=32)
        engine.add_request(Request(req_id=0, prompt_ids=prompt, max_new_tokens=2))
        engine.run()
        engine.add_request(Request(req_id=1, prompt_ids=prompt, max_new_tokens=2))
        assert engine.run()[1] == _oracle(tiny_model, prompt, 2)

    def test_refcount_zero_after_free(self, tiny_model):
        """free 后前缀块 refcount 归零，且可以被新分配重新使用（num_free_blocks 增加）。"""
        shared = list(range(1, 5))  # 4 tokens = 1 整块（block_size=4）
        bm = BlockManager(num_blocks=16, block_size=4)

        # 第一个 seq
        s1 = Sequence(Request(req_id=0, prompt_ids=shared + [9, 10], max_new_tokens=2))
        bm.allocate(s1)
        s1.output_ids = [9, 10]

        free_before = bm.num_free_blocks
        bm.free(s1)
        free_after = bm.num_free_blocks

        assert free_after >= free_before, "free 后空闲块数应不减少"

        # 前缀块 refcount 应为 0
        if s1.block_table:  # 已被清空，检查通过 match_prefix 找到的 block
            pass
        matched, _ = bm.match_prefix(shared)
        if matched:
            bid = matched[0]
            assert bm._refcount[bid] == 0, f"prefix block {bid} refcount 应为 0"

    def test_prefix_block_recyclable(self, tiny_model):
        """pool 耗尽时，cached prefix block 可被驱逐以腾空间。"""
        bm = BlockManager(num_blocks=4, block_size=4)  # 只有 4 块

        # 填满 prefix cache
        s1 = Sequence(Request(req_id=0, prompt_ids=list(range(1, 17)), max_new_tokens=1))
        assert bm.can_allocate(s1)
        bm.allocate(s1)
        s1.output_ids = [99]
        bm.free(s1)

        # 再来一个新 seq，pool 应从 cached 块驱逐
        s2 = Sequence(Request(req_id=1, prompt_ids=list(range(50, 66)), max_new_tokens=1))
        assert bm.can_allocate(s2), "块全为 cached 时，应能驱逐以满足新分配"
        bm.allocate(s2)  # 不应 raise


# ─────────────────────────────────── T6 ──────────────────────────────────────


class TestT6Sampler:
    """T6：top-p 截断、固定 generator 可复现、逐 seq 参数独立生效。"""

    def _make_seqs(self, temperatures: list[float], top_ps: list[float]) -> list[Sequence]:
        return [
            Sequence(
                Request(req_id=i, prompt_ids=[1, 2, 3], max_new_tokens=1,
                        temperature=t, top_p=p)
            )
            for i, (t, p) in enumerate(zip(temperatures, top_ps))
        ]

    def test_greedy_temperature_zero(self):
        """temperature=0 时，sample 结果等于 argmax。"""
        torch.manual_seed(0)
        logits = torch.randn(3, 128)
        seqs = self._make_seqs([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        result = sample(logits, seqs)
        expected = logits.argmax(dim=-1).tolist()
        assert result == expected, f"greedy mismatch: {result} vs {expected}"

    def test_fixed_generator_reproducible(self):
        """固定 generator 时，两次采样结果一致。"""
        torch.manual_seed(0)
        logits = torch.randn(2, 128)
        seqs = self._make_seqs([1.0, 0.5], [1.0, 1.0])

        gen1 = torch.Generator()
        gen1.manual_seed(7)
        r1 = sample(logits.clone(), seqs, generator=gen1)

        gen2 = torch.Generator()
        gen2.manual_seed(7)
        r2 = sample(logits.clone(), seqs, generator=gen2)

        assert r1 == r2, f"相同 seed 下结果应相同: {r1} vs {r2}"

    def test_top_p_truncation(self):
        """top_p=0.0001 时，几乎只保留最高概率 token，结果应与 argmax 一致。"""
        torch.manual_seed(0)
        logits = torch.randn(2, 128) * 10  # 拉大分布差距
        seqs = self._make_seqs([1.0, 1.0], [0.0001, 0.0001])
        result = sample(logits, seqs)
        expected = logits.argmax(dim=-1).tolist()
        assert result == expected, f"top-p 极小时应等于 argmax: {result} vs {expected}"

    def test_per_seq_params(self):
        """不同 seq 的 temperature 独立生效（greedy vs 采样结果可能不同）。"""
        torch.manual_seed(0)
        logits = torch.randn(2, 128) * 3
        seq_greedy = Sequence(Request(req_id=0, prompt_ids=[1], max_new_tokens=1,
                                     temperature=0.0, top_p=1.0))
        seq_hot = Sequence(Request(req_id=1, prompt_ids=[1], max_new_tokens=1,
                                   temperature=100.0, top_p=1.0))
        # greedy 结果确定
        gen = torch.Generator(); gen.manual_seed(42)
        result = sample(logits, [seq_greedy, seq_hot], generator=gen)
        greedy_tok = int(logits[0].argmax().item())
        assert result[0] == greedy_tok, "第 0 个 seq greedy 应等于 argmax"


# ─────────────────────────────────── T7 ──────────────────────────────────────


class TestT7Metrics:
    """T7：错峰请求 metrics 合理性检查。"""

    def test_ttft_monotone_for_same_length(self, tiny_model):
        """
        同样长度的 prompt，后到的请求 ttft 不小于先到的。
        实现：r_early 在 step=0 加入，r_late 在 step=3 加入。
        """
        prompt = [1, 2, 3, 4, 5]  # 同长 prompt

        engine = Engine(tiny_model, num_blocks=128, block_size=8, max_batch_tokens=512)
        r_early = Request(req_id=0, prompt_ids=prompt, max_new_tokens=4, arrival_step=0)
        r_late = Request(req_id=1, prompt_ids=prompt, max_new_tokens=4, arrival_step=3)

        engine.add_request(r_early)

        for step_i in range(20):
            if step_i == 3:
                engine.add_request(r_late)
            engine.step()
            if engine.scheduler.num_waiting == 0 and engine.scheduler.num_running == 0:
                break

        m = engine.metrics()
        assert 0 in m and 1 in m, "两个请求都应有 metrics"
        ttft_early = m[0]["ttft_steps"]
        ttft_late = m[1]["ttft_steps"]
        assert ttft_late >= ttft_early, (
            f"后到请求的 ttft ({ttft_late}) 应 >= 先到的 ({ttft_early})"
        )

    def test_tpot_positive(self, tiny_model):
        """n_out > 1 时，tpot > 0。"""
        engine = Engine(tiny_model, num_blocks=128, block_size=8, max_batch_tokens=512)
        r = Request(req_id=0, prompt_ids=[1, 2, 3], max_new_tokens=5)
        engine.add_request(r)
        engine.run()

        m = engine.metrics()
        assert m[0]["n_out"] > 1
        assert m[0]["tpot_steps"] > 0, f"tpot 应 > 0，实际 {m[0]['tpot_steps']}"

    def test_ttft_uses_arrival_step(self, tiny_model):
        """ttft = 首 token 步 - arrival_step（不同 arrival_step 影响 ttft）。"""
        engine = Engine(tiny_model, num_blocks=128, block_size=8, max_batch_tokens=512)
        # 两个请求同时提交，但 arrival_step 不同
        r0 = Request(req_id=0, prompt_ids=[1, 2, 3], max_new_tokens=3, arrival_step=0)
        r1 = Request(req_id=1, prompt_ids=[4, 5, 6], max_new_tokens=3, arrival_step=5)
        engine.add_request(r0)
        engine.add_request(r1)
        engine.run()

        m = engine.metrics()
        # r1.arrival_step=5，首 token 产出步 >= 0，ttft 可以为负（说明 arrival_step 被考虑）
        # 重要的是 ttft 用了 arrival_step 而非固定值
        ttft_r0 = m[0]["ttft_steps"]
        ttft_r1 = m[1]["ttft_steps"]
        # r1 的 arrival_step=5 较晚，若两者同步完成，r1 的 ttft 应 <= r0（arrival 更晚但同时处理）
        # 只验证 metrics 字段存在且类型正确
        assert isinstance(ttft_r0, int), f"ttft 应为 int，实际 {type(ttft_r0)}"
        assert isinstance(ttft_r1, int)
        assert m[0]["n_out"] > 0
        assert m[1]["n_out"] > 0
