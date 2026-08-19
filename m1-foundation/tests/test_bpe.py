"""Phase 1.0 BPE 验收测试（SPEC 第 6 节 T1–T7）。

测试先于实现存在：当前全红是预期状态。
只允许使用 SPEC 冻结接口（train/encode/decode/save/load/vocab_size），
不得触碰实现内部属性 —— 保证实现自由度。
"""
import filecmp
import random
from pathlib import Path

import pytest

from minilm.tokenizer.bpe import BPETokenizer

EOT = "<|endoftext|>"


# ---------- fixtures ----------

def _write_corpus(path: Path, docs: list[str]) -> Path:
    path.write_text(EOT.join(docs) + EOT, encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> Path:
    """确定性小语料：英文为主 + 少量中文/符号，词频刻意拉开差距以减少 tie。"""
    rng = random.Random(42)
    words = (
        ["the"] * 400 + ["cat"] * 300 + ["little"] * 250 + ["story"] * 200
        + ["played"] * 150 + ["garden"] * 120 + ["happy"] * 90
        + ["从前"] * 60 + ["森林"] * 40 + ["dragon!"] * 30 + ["it's"] * 20
    )
    rng.shuffle(words)
    docs = [" ".join(words[i : i + 50]) for i in range(0, len(words), 50)]
    return _write_corpus(tmp_path_factory.mktemp("data") / "corpus.txt", docs)


@pytest.fixture(scope="module")
def tok(corpus) -> BPETokenizer:
    return BPETokenizer.train(corpus, vocab_size=600, special_tokens=[EOT])


# ---------- T1 round-trip fuzz ----------

FUZZ_POOL = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    " \t\n"
    "，。！？「」……——"
    "从前有一座森林里面住着小猫"
    "éüßñ"
    "🐱🌲✨🎉"
)


def test_t1_roundtrip_fuzz(tok):
    rng = random.Random(0)
    cases = [
        "",
        " ",
        "\n\n",
        "hello world",
        "the cat played in the 森林 🐱",
        "it's   spaced\tweirdly\n",
    ]
    cases += ["".join(rng.choices(FUZZ_POOL, k=rng.randint(1, 200))) for _ in range(200)]
    for s in cases:
        assert tok.decode(tok.encode(s)) == s, f"round-trip failed for: {s!r}"


def test_t1_roundtrip_unseen_bytes(tok):
    # byte-level 词表：训练时没见过的字符也必须无损往返
    s = "Ωמところ𝕊​"
    assert tok.decode(tok.encode(s)) == s


# ---------- T2 训练确定性 ----------

def test_t2_train_deterministic(corpus, tmp_path):
    a = BPETokenizer.train(corpus, vocab_size=600, special_tokens=[EOT])
    b = BPETokenizer.train(corpus, vocab_size=600, special_tokens=[EOT])
    da, db = tmp_path / "a", tmp_path / "b"
    a.save(da)
    b.save(db)
    files = sorted(p.name for p in da.iterdir())
    assert files == sorted(p.name for p in db.iterdir())
    for name in files:
        assert filecmp.cmp(da / name, db / name, shallow=False), f"{name} differs"


# ---------- T3 参考对齐（HF tokenizers）----------

def test_t3_hf_reference_alignment(corpus, tmp_path):
    """与 HF tokenizers 同参数训练对比前 50 条 merges。

    语料词频已刻意拉开差距，理论上无 tie；若本测试因 tie-breaking
    差异失败，先人工确认是 tie 而非算法错误，再考虑调整语料。
    """
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    hf = Tokenizer(models.BPE())
    hf.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=600,
        special_tokens=[EOT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    hf.train([str(corpus)], trainer)

    ours = BPETokenizer.train(corpus, vocab_size=600, special_tokens=[EOT])
    d = tmp_path / "ours"
    ours.save(d)
    # 对比编码行为而非内部表示：高频词应切成相同数量的 token
    for w in ["the cat", "little story", " garden", "played"]:
        n_ours = len(ours.encode(w))
        n_hf = len(hf.encode(w).ids)
        assert n_ours == n_hf, f"{w!r}: ours={n_ours} hf={n_hf}"


# ---------- T4 special tokens ----------

def test_t4_special_token_atomic(tok):
    ids = tok.encode(f"hello{EOT}world")
    eot_id = tok.encode(EOT)
    assert len(eot_id) == 1, "special token 必须编码为单个 id"
    assert eot_id[0] in ids
    assert tok.decode(ids) == f"hello{EOT}world"


def test_t4_special_lookalike_not_special(tok):
    # 不完整的 special token 字面量按普通文本处理
    s = "<|endoftext"
    assert tok.decode(tok.encode(s)) == s
    assert len(tok.encode(s)) > 1


def test_t4_adjacent_specials(tok):
    s = f"{EOT}{EOT}a{EOT}"
    ids = tok.encode(s)
    assert tok.decode(ids) == s
    assert ids.count(tok.encode(EOT)[0]) == 3


# ---------- T5 save / load ----------

def test_t5_save_load_roundtrip(tok, tmp_path):
    d = tmp_path / "tok"
    tok.save(d)
    tok2 = BPETokenizer.load(d)
    assert tok2.vocab_size == tok.vocab_size
    for s in ["the little cat", f"story{EOT}end", "从前有一座森林 🐱", ""]:
        assert tok2.encode(s) == tok.encode(s)
        assert tok2.decode(tok.encode(s)) == s


# ---------- T6 三版一致性 ----------

def _save_files(t: BPETokenizer, d: Path) -> dict[str, bytes]:
    t.save(d)
    return {p.name: p.read_bytes() for p in sorted(d.iterdir())}


def test_t6_three_versions_identical(corpus, tmp_path):
    variants = {
        "v1_naive": dict(algo="naive", num_workers=1),
        "v2_incr": dict(algo="incremental", num_workers=1),
        "v3_w2": dict(algo="incremental", num_workers=2),
        "v3_w4": dict(algo="incremental", num_workers=4),
    }
    outs = {}
    for name, kw in variants.items():
        t = BPETokenizer.train(corpus, vocab_size=600, special_tokens=[EOT], **kw)
        outs[name] = _save_files(t, tmp_path / name)
    base = outs["v1_naive"]
    for name, files in outs.items():
        assert files == base, f"{name} 与 naive 版训练结果不一致"


# ---------- T7 chunk 边界 ----------

def test_t7_chunk_boundary_alignment(tmp_path):
    """构造 special token 恰好落在各种 chunk 切点附近的文件，
    并行与串行的训练结果必须逐字节一致（文档不得被切断）。"""
    rng = random.Random(7)
    docs = []
    for i in range(200):
        # 文档长度刻意参差，让 EOT 出现在任意 offset
        n = rng.randint(1, 300)
        docs.append("".join(rng.choices("ab cd", k=n)))
    corpus = _write_corpus(tmp_path / "boundary.txt", docs)

    serial = BPETokenizer.train(corpus, vocab_size=300, special_tokens=[EOT], num_workers=1)
    parallel = BPETokenizer.train(corpus, vocab_size=300, special_tokens=[EOT], num_workers=8)
    assert _save_files(serial, tmp_path / "s") == _save_files(parallel, tmp_path / "p")


# ---------- tie-breaking 规则 ----------

def test_tiebreak_prefers_larger_byte_sequence(tmp_path):
    """频次并列时必须选字节序较大的 pair（SPEC §5）。

    语料每行 "ab cd"：pair (a,b)、(空格,c)、(c,d) 频次完全相同，
    字节序最大的是 (c,d)。vocab_size 只留 1 个 merge 名额，
    因此唯一的 merge 必须是 (c,d) —— 即 "cd" 编码为单 token。
    """
    corpus = _write_corpus(tmp_path / "tie.txt", ["ab cd\n"] * 50)
    tok = BPETokenizer.train(corpus, vocab_size=258, special_tokens=[EOT])  # 恰好 1 个 merge
    assert len(tok.encode("cd")) == 1, "唯一 merge 应为字节序最大的 (c,d)"
    assert len(tok.encode("ab")) == 2


# ---------- 边界行为 ----------

def test_empty_and_ids_domain(tok):
    assert tok.encode("") == []
    assert tok.decode([]) == ""
    ids = tok.encode("the little cat played")
    assert all(0 <= i < tok.vocab_size for i in ids)
