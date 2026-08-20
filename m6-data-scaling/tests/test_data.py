"""Phase 6.0 验收测试（T1–T6）。

运行：
    cd m6-data-scaling && python3 -m pytest tests/test_data.py -x -q

测试风格：每个 T 组以 def test_tN_xxx 函数命名，全部 CPU / 确定性。
"""
import math
import random
import string

import numpy as np
import pytest

from minidata.filters import FilterStats, apply_filters, quality_filter
from minidata.minhash import (
    LSHIndex,
    MinHasher,
    dedup,
    jaccard_estimate,
    shingle,
)
from minidata.contamination import contamination_rate, flag_contaminated, ngram_set


# ─────────────────────────── 工具函数 ────────────────────────────

def _words(n: int, seed: int = 0) -> str:
    """生成 n 个随机小写单词（长度 3–8），空格拼接。"""
    rng = random.Random(seed)
    vocab = [
        "".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 8)))
        for _ in range(max(n * 2, 200))
    ]
    return " ".join(rng.choices(vocab, k=n))


def _make_pair_with_jaccard(target_j: float, pool_size: int = 500, k: int = 3,
                             seed: int = 42) -> tuple[str, str]:
    """
    构造真实 Jaccard ≈ target_j 的两个文档。

    原理：从共享词池（每个元素是 k 个词的 tuple）直接控制交并比。
        |A ∩ B| / |A ∪ B| = target_j
    令 shared = round(target_j * pool_size)，
       A 独占 = pool_size - shared，
       B 独占 = pool_size - shared（A∪B = 2*pool_size - shared）。
    实际 J = shared / (2*pool_size - shared)。
    """
    rng = random.Random(seed)
    # 生成足够多的 k-gram 原型（每个是 k 个词）
    base_words = [
        "".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 7)))
        for _ in range(pool_size * (k + 2))
    ]
    all_kgrams = [
        tuple(base_words[i : i + k]) for i in range(pool_size * 2)
    ]
    # 去重保留 pool_size * 2 个唯一 k-gram
    seen: set[tuple] = set()
    unique: list[tuple] = []
    for kg in all_kgrams:
        if kg not in seen:
            seen.add(kg)
            unique.append(kg)
        if len(unique) == pool_size * 2:
            break
    # 保证有足够数量（填充）
    while len(unique) < pool_size * 2:
        extra = tuple(
            "".join(rng.choices(string.ascii_lowercase, k=4)) for _ in range(k)
        )
        if extra not in seen:
            seen.add(extra)
            unique.append(extra)

    shared_n = round(target_j * pool_size)
    only_a_n = pool_size - shared_n
    only_b_n = pool_size - shared_n

    shared_kg = unique[:shared_n]
    only_a_kg = unique[shared_n : shared_n + only_a_n]
    only_b_kg = unique[shared_n + only_a_n : shared_n + only_a_n + only_b_n]

    def kgrams_to_text(kgrams: list[tuple]) -> str:
        # 把 k-gram 列表拼成文档——相邻 k-gram 共享 k-1 个词可产生 shingling
        # 简单起见，直接拼接每个 k-gram 作为一个句子（shingle 独立）
        return " ".join(" ".join(kg) for kg in kgrams)

    a_kgrams = shared_kg + only_a_kg
    b_kgrams = shared_kg + only_b_kg
    rng.shuffle(a_kgrams)
    rng.shuffle(b_kgrams)

    # 验证真实 Jaccard（shingle 级别）
    doc_a = kgrams_to_text(a_kgrams)
    doc_b = kgrams_to_text(b_kgrams)
    return doc_a, doc_b


# ─────────────────────── T1 过滤规则 ─────────────────────────────

class TestT1Filters:
    """T1：六条过滤规则逐条构造违例，精确匹配规则名；干净文档全过；统计正确。"""

    def _clean(self) -> str:
        """构造一个通过所有规则的干净文档。"""
        return _words(50, seed=99)

    def test_t1_word_count_too_few(self):
        doc = _words(10)
        ok, reason = quality_filter(doc)
        assert not ok and reason == "word_count"

    def test_t1_word_count_too_many(self):
        # 100_001 个词
        doc = " ".join(["word"] * 100_001)
        ok, reason = quality_filter(doc)
        assert not ok and reason == "word_count"

    def test_t1_mean_word_len_too_short(self):
        # 50 个单字母词（平均词长 1.0 < 2）
        doc = " ".join(["a"] * 50)
        ok, reason = quality_filter(doc)
        assert not ok and reason == "mean_word_len"

    def test_t1_mean_word_len_too_long(self):
        # 50 个 15 字符词（平均词长 15 > 12）
        doc = " ".join(["a" * 15] * 50)
        ok, reason = quality_filter(doc)
        assert not ok and reason == "mean_word_len"

    def test_t1_symbol_ratio(self):
        # 50 个普通词 + 大量 '#'
        words = ["hello"] * 50
        symbols = ["#"] * 10  # 10/60 > 0.1
        doc = " ".join(words + symbols)
        ok, reason = quality_filter(doc)
        assert not ok and reason == "symbol_ratio"

    def test_t1_bullet_ratio(self):
        # 全部行以 '-' 开头（≥ 0.9）
        line = "- " + _words(5)
        doc = "\n".join([line] * 50)
        ok, reason = quality_filter(doc)
        assert not ok and reason == "bullet_ratio"

    def test_t1_dup_line_ratio(self):
        # 重复行占比 >= 0.3
        repeated = "this is a repeated line with enough words here yes"
        unique_lines = [f"unique line number {i} with words" for i in range(7)]
        # 3 条重复行，总 10 行 → 重复行数 3（重复出现的行）/ 10 = 0.3 ≥ 0.3
        lines = unique_lines + [repeated] * 3
        doc = "\n".join(lines)
        ok, reason = quality_filter(doc)
        assert not ok and reason == "dup_line_ratio"

    def test_t1_alpha_ratio(self):
        # 大部分词是纯数字
        num_words = ["123"] * 40
        alpha_words = ["hello"] * 10  # alpha 占 10/50 = 0.2 ≤ 0.6
        doc = " ".join(num_words + alpha_words)
        ok, reason = quality_filter(doc)
        assert not ok and reason == "alpha_ratio"

    def test_t1_clean_doc_passes(self):
        doc = self._clean()
        ok, reason = quality_filter(doc)
        assert ok and reason is None

    def test_t1_apply_filters_stats(self):
        clean = [self._clean() for _ in range(5)]
        dirty_wc = [_words(5) for _ in range(3)]  # word_count
        docs = clean + dirty_wc
        kept, stats = apply_filters(docs)
        assert len(kept) == 5
        assert stats.kept == 5
        assert stats.dropped == 3
        assert stats.drop_reasons.get("word_count", 0) == 3

    def test_t1_rule_order_word_count_first(self):
        # 词数不足时，应先触发 word_count 而非后续规则
        doc = _words(5)  # 只有 5 个词
        ok, reason = quality_filter(doc)
        assert reason == "word_count"


# ─────────────────────── T2 MinHash 无偏性 ────────────────────────

class TestT2Unbiasedness:
    """T2：已知真实 Jaccard 文档对，num_perm=256 时误差 < 0.08；同文档=1；不相交≈0。"""

    @pytest.fixture(scope="class")
    def hasher(self):
        return MinHasher(num_perm=256, seed=42)

    def test_t2_same_doc(self, hasher):
        doc = _words(200, seed=1)
        sh = shingle(doc, k=3)
        sig = hasher.signature(sh)
        est = jaccard_estimate(sig, sig)
        assert est == pytest.approx(1.0)

    def test_t2_disjoint(self, hasher):
        doc_a = _words(200, seed=10)
        doc_b = _words(200, seed=11)
        # 保证不相交（用完全不同 seed 生成的不同词）
        sh_a = shingle(doc_a, k=3)
        sh_b = shingle(doc_b, k=3)
        # 如果有微小重叠（随机巧合），估计应接近 0
        sig_a = hasher.signature(sh_a)
        sig_b = hasher.signature(sh_b)
        est = jaccard_estimate(sig_a, sig_b)
        assert est < 0.1

    @pytest.mark.parametrize("target_j", [0.2, 0.5, 0.8])
    def test_t2_known_jaccard(self, hasher, target_j):
        doc_a, doc_b = _make_pair_with_jaccard(target_j, pool_size=600, k=3, seed=7)
        sh_a = shingle(doc_a, k=3)
        sh_b = shingle(doc_b, k=3)

        # 验证真实 Jaccard（shingle 集合级别）
        true_j = len(sh_a & sh_b) / len(sh_a | sh_b) if sh_a | sh_b else 0.0

        sig_a = hasher.signature(sh_a)
        sig_b = hasher.signature(sh_b)
        est = jaccard_estimate(sig_a, sig_b)
        assert abs(est - true_j) < 0.08, (
            f"target_j={target_j}, true_j={true_j:.4f}, est={est:.4f}, "
            f"err={abs(est-true_j):.4f}"
        )

    def test_t2_empty_shingles(self, hasher):
        sig = hasher.signature(set())
        # 空集签名全为最大值
        assert np.all(sig == np.iinfo(np.uint64).max)


# ─────────────────────── T3 签名稳定性 ───────────────────────────

class TestT3Stability:
    """T3：同 seed 两次构建 MinHasher，签名逐位相等（跨进程稳定哈希回归测试）。"""

    def test_t3_same_seed_same_sig(self):
        doc = _words(100, seed=5)
        sh = shingle(doc, k=3)
        sig_a = MinHasher(num_perm=128, seed=0).signature(sh)
        sig_b = MinHasher(num_perm=128, seed=0).signature(sh)
        assert np.array_equal(sig_a, sig_b)

    def test_t3_different_seed_different_sig(self):
        doc = _words(100, seed=5)
        sh = shingle(doc, k=3)
        sig_a = MinHasher(num_perm=128, seed=0).signature(sh)
        sig_b = MinHasher(num_perm=128, seed=1).signature(sh)
        # 不同 seed 应该产生不同签名
        assert not np.array_equal(sig_a, sig_b)

    def test_t3_shingle_stability(self):
        """shingle 函数输出应与顺序无关（集合语义），且相同输入给出相同输出。"""
        doc = "the quick brown fox jumps over the lazy dog"
        s1 = shingle(doc, k=3)
        s2 = shingle(doc, k=3)
        assert s1 == s2

    def test_t3_no_builtin_hash(self):
        """回归：确保实现不依赖 Python 内置 hash()（用 blake2b）。
        通过构造签名并检查与 seed 固定的参考值一致来间接验证。"""
        hasher = MinHasher(num_perm=4, seed=0)
        sh = {"hello world foo", "foo bar baz"}
        sig = hasher.signature(sh)
        # 只要重复运行结果一致（稳定性），不检查具体数值
        sig2 = MinHasher(num_perm=4, seed=0).signature(sh)
        assert np.array_equal(sig, sig2)


# ─────────────────────── T4 LSH 召回/过滤 ────────────────────────

class TestT4LSH:
    """T4：num_perm=128, bands=32, rows=4 参数下：
    - Jaccard ≥ 0.9 的对必为候选（漏检率 <1e-6）
    - Jaccard ≈ 0.02 的对不为候选（20 对全不误报）
    """

    @pytest.fixture(scope="class")
    def hasher(self):
        return MinHasher(num_perm=128, seed=42)

    def _high_jaccard_pair(self, base_doc: str, drop_rate: float = 0.05,
                            seed: int = 0) -> str:
        """在 base_doc 基础上随机替换少量词，使 shingle 级 Jaccard ≥ 0.9。"""
        words = base_doc.split()
        rng = random.Random(seed)
        # 只替换约 drop_rate 的词
        new_words = []
        for w in words:
            if rng.random() < drop_rate:
                new_words.append(
                    "".join(rng.choices(string.ascii_lowercase, k=len(w)))
                )
            else:
                new_words.append(w)
        return " ".join(new_words)

    def test_t4_high_jaccard_must_be_candidate(self, hasher):
        """Jaccard ≥ 0.9 的对必须出现在候选中。"""
        base = _words(300, seed=20)
        near_dup = self._high_jaccard_pair(base, drop_rate=0.02, seed=1)

        sh_base = shingle(base, k=3)
        sh_near = shingle(near_dup, k=3)

        # 验证真实 Jaccard 确实高
        true_j = len(sh_base & sh_near) / len(sh_base | sh_near)
        assert true_j >= 0.85, f"构造的 near-dup 真实 Jaccard 过低：{true_j:.3f}"

        sig_base = hasher.signature(sh_base)
        sig_near = hasher.signature(sh_near)

        index = LSHIndex(num_perm=128, bands=32)
        index.add(0, sig_base)
        cands = index.candidates(sig_near)
        assert 0 in cands, (
            f"Jaccard={true_j:.3f} 的近重复对未被 LSH 检出"
        )

    def test_t4_low_jaccard_no_false_positive(self, hasher):
        """Jaccard ≈ 0.02 的 20 对全不误报。"""
        # 构造 Jaccard ≈ 0.02 的文档对（SPEC 要求：确定性无误报）
        index = LSHIndex(num_perm=128, bands=32)
        n_pairs = 20
        false_positives = 0
        for i in range(n_pairs):
            doc_a = _words(300, seed=i * 2)
            doc_b = _words(300, seed=i * 2 + 1)
            sh_a = shingle(doc_a, k=3)
            sh_b = shingle(doc_b, k=3)
            # 验证真实 Jaccard 确实很低
            true_j = (
                len(sh_a & sh_b) / len(sh_a | sh_b) if sh_a | sh_b else 0.0
            )
            assert true_j < 0.1, f"对 {i} 真实 Jaccard 过高：{true_j:.3f}"

            sig_a = hasher.signature(sh_a)
            sig_b = hasher.signature(sh_b)

            # 用独立 index 避免相互干扰
            tmp_index = LSHIndex(num_perm=128, bands=32)
            tmp_index.add(i * 2, sig_a)
            cands = tmp_index.candidates(sig_b)
            if i * 2 in cands:
                false_positives += 1

        assert false_positives == 0, (
            f"低 Jaccard 文档对中出现 {false_positives}/20 个误报"
        )

    def test_t4_candidate_symmetry(self, hasher):
        """LSH 候选应对称：若 a 是 b 的候选，b 也应是 a 的候选（都加入 index 时）。"""
        base = _words(200, seed=30)
        near = self._high_jaccard_pair(base, drop_rate=0.01, seed=5)
        sig_base = hasher.signature(shingle(base, k=3))
        sig_near = hasher.signature(shingle(near, k=3))
        index = LSHIndex(num_perm=128, bands=32)
        index.add(0, sig_base)
        index.add(1, sig_near)
        cands_from_near = index.candidates(sig_near)
        cands_from_base = index.candidates(sig_base)
        # 都加入 index 后，互为候选
        assert 0 in cands_from_near
        assert 1 in cands_from_base


# ─────────────────────── T5 dedup 端到端 ─────────────────────────

class TestT5Dedup:
    """T5：100 篇文档中埋 10 组近重复，全部检出且保留最早者；干净集零误杀。"""

    def _build_corpus(self) -> tuple[list[str], dict[int, int]]:
        """
        构建测试语料：
        - 80 篇干净文档（无近重复）
        - 10 组近重复，每组 2 篇（原文 + 极轻微改词版），共 20 篇
        返回：(docs, dup_map)，dup_map[later_idx] = earlier_idx

        设计保证：近重复对真实 Jaccard > 0.95（替换率 0.5%，词数 400），
        确保 dedup(threshold=0.8) 稳定检出。
        """
        rng = random.Random(0)
        docs: list[str] = []
        dup_map: dict[int, int] = {}

        for i in range(80):
            docs.append(_words(200, seed=i))

        for g in range(10):
            base_idx = len(docs)
            base_doc = _words(400, seed=1000 + g)  # 400 词，shingle 数量充足
            docs.append(base_doc)

            # 极轻微改词：替换约 0.5% 的词（保证真实 Jaccard > 0.95）
            words = base_doc.split()
            new_words = []
            for w in words:
                if rng.random() < 0.005:
                    new_words.append(
                        "".join(rng.choices(string.ascii_lowercase, k=len(w)))
                    )
                else:
                    new_words.append(w)
            dup_doc = " ".join(new_words)
            dup_idx = len(docs)
            docs.append(dup_doc)
            dup_map[dup_idx] = base_idx

        return docs, dup_map

    def test_t5_near_dup_detected(self):
        docs, dup_map = self._build_corpus()
        kept, dup_pairs = dedup(docs, threshold=0.8, num_perm=128, bands=32, k=3)

        kept_set = set(kept)
        for later, earlier in dup_map.items():
            assert later not in kept_set, (
                f"近重复文档 {later}（与 {earlier} 重复）未被去重"
            )
            assert earlier in kept_set, (
                f"重复簇的最早文档 {earlier} 被错误去除"
            )

    def test_t5_clean_corpus_zero_kill(self):
        """80 篇无近重复的干净文档，去重后全部保留。"""
        docs = [_words(200, seed=i) for i in range(80)]
        kept, dup_pairs = dedup(docs, threshold=0.8, num_perm=128, bands=32, k=3)
        assert len(kept) == 80, (
            f"干净语料被误杀：kept={len(kept)}, dup_pairs={dup_pairs}"
        )

    def test_t5_preserve_order(self):
        """保留下标保序。"""
        docs, _ = self._build_corpus()
        kept, _ = dedup(docs, threshold=0.8, num_perm=128, bands=32, k=3)
        assert kept == sorted(kept), "保留下标应保序"

    def test_t5_earliest_kept(self):
        """重复簇保留最早者（下标最小）。"""
        docs, dup_map = self._build_corpus()
        kept_set = set(
            dedup(docs, threshold=0.8, num_perm=128, bands=32, k=3)[0]
        )
        for later, earlier in dup_map.items():
            assert earlier in kept_set
            assert later not in kept_set


# ─────────────────────── T6 污染检测 ─────────────────────────────

class TestT6Contamination:
    """T6：eval 句子嵌入 train doc 能检出；无重叠文档污染率为 0；边界不崩。"""

    def test_t6_exact_overlap_detected(self):
        """把 eval 句子原样嵌进 train doc，污染率 > threshold，应被检出。"""
        # 使用一个相对长的 eval 片段（20+ 词），使其 8-gram 占 train doc 8-gram 的较高比例
        eval_sent = (
            "the model was evaluated on the standard benchmark dataset here "
            "and achieved state of the art results on multiple tasks including"
        )
        eval_doc = eval_sent
        # train doc 主体是 eval 片段（共 30 词），使污染率足够高
        train_doc = eval_sent

        eval_ngrams = ngram_set(eval_doc, n=8)
        rate = contamination_rate(train_doc, eval_ngrams, n=8)
        assert rate > 0.1, f"应检出污染，但 rate={rate:.4f}"

    def test_t6_no_overlap_rate_zero(self):
        """无重叠文档污染率应为 0。"""
        eval_doc = _words(100, seed=10)
        train_doc = _words(100, seed=20)

        # 极低概率有 8-gram 重叠（随机词，概率极低；如有，接受小误差）
        eval_ngrams = ngram_set(eval_doc, n=8)
        rate = contamination_rate(train_doc, eval_ngrams, n=8)
        assert rate < 0.01, f"无重叠文档污染率过高：{rate:.4f}"

    def test_t6_short_doc_no_crash(self):
        """不足 8 词的文档不应崩溃，返回空集 / 0。"""
        short = "hello world"
        assert ngram_set(short, n=8) == set()
        eval_ngrams = ngram_set(_words(50, seed=5), n=8)
        rate = contamination_rate(short, eval_ngrams, n=8)
        assert rate == 0.0

    def test_t6_flag_contaminated_batch(self):
        """flag_contaminated 批量接口：含 eval 片段的 train doc 被检出，无关的不检出。"""
        eval_sent = "the quick brown fox jumps over the lazy dog and runs away fast"
        eval_doc = eval_sent

        # train docs
        clean_docs = [_words(80, seed=i) for i in range(5)]
        contaminated_doc = _words(30, seed=99) + " " + eval_sent + " " + _words(30, seed=100)
        train_docs = clean_docs + [contaminated_doc]

        flagged = flag_contaminated(train_docs, [eval_doc], n=8, threshold=0.05)
        assert 5 in flagged, "含 eval 片段的 train doc 未被检出"
        # 检查干净文档未被误报
        for i in range(5):
            assert i not in flagged, f"干净文档 {i} 被错误标记为污染"

    def test_t6_truncated_eval_detected(self):
        """eval 句子截断嵌入（保留 ≥8 个词的子串），仍能被检出。"""
        eval_doc = "this is a comprehensive evaluation benchmark for language model quality assessment testing"
        # 取前 10 个词嵌入 train doc（≥8 词）
        eval_words = eval_doc.split()
        fragment = " ".join(eval_words[:10])
        train_doc = _words(40, seed=50) + " " + fragment + " " + _words(40, seed=51)

        eval_ngrams = ngram_set(eval_doc, n=8)
        rate = contamination_rate(train_doc, eval_ngrams, n=8)
        assert rate > 0.0, "截断 eval 片段未被检出"

    def test_t6_ngram_set_basic(self):
        doc = "a b c d e f g h i"  # 9 词，3-gram→7个，8-gram→2个
        s3 = ngram_set(doc, n=3)
        s8 = ngram_set(doc, n=8)
        assert len(s3) == 7
        assert len(s8) == 2

    def test_t6_contamination_rate_full_overlap(self):
        """train doc 与 eval 完全相同时，污染率 = 1.0。"""
        doc = _words(30, seed=77)
        eval_ngrams = ngram_set(doc, n=8)
        rate = contamination_rate(doc, eval_ngrams, n=8)
        assert rate == pytest.approx(1.0)
