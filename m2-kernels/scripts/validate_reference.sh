#!/usr/bin/env bash
# validate_reference.sh
# 第一次租卡时运行此脚本：
#   1. 把参考答案覆盖到骨架位置
#   2. 跑全部测试（应全绿）
#   3. 恢复骨架
# 全绿 → 出题包质量 OK，可以开始闯关；
# 有红 → 是出题人的 bug，先报给出题人修，不要开始闯关。
#
# 用法：
#   bash scripts/validate_reference.sh
# 确保工作目录在 m2-kernels/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

cd "$ROOT"

KERNELS="$ROOT/kernels"
REF="$ROOT/reference/2.0-kernels"
BACKUP="$ROOT/.backup_skeletons"

echo "=== Phase 2.0 参考答案验证脚本 ==="
echo "工作目录：$ROOT"
echo

# ─── 检查前置条件 ─────────────────────────────────────────────────────────────
if ! python3 -c "import torch; assert torch.cuda.is_available(), 'No CUDA'" 2>/dev/null; then
    echo "错误：CUDA GPU 不可用。请在有 NVIDIA GPU 的机器上运行。"
    exit 1
fi

if ! python3 -c "import triton" 2>/dev/null; then
    echo "错误：triton 未安装。请先安装：pip install triton"
    exit 1
fi

# ─── 备份骨架 ─────────────────────────────────────────────────────────────────
echo "[1/4] 备份学生骨架..."
mkdir -p "$BACKUP"
cp "$KERNELS/rmsnorm.py"      "$BACKUP/rmsnorm.py"
cp "$KERNELS/swiglu.py"       "$BACKUP/swiglu.py"
cp "$KERNELS/cross_entropy.py" "$BACKUP/cross_entropy.py"
echo "      备份至 $BACKUP/"

# ─── 覆盖参考答案 ─────────────────────────────────────────────────────────────
echo "[2/4] 覆盖参考答案..."
cp "$REF/rmsnorm_solution.py"      "$KERNELS/rmsnorm.py"
cp "$REF/swiglu_solution.py"       "$KERNELS/swiglu.py"
cp "$REF/cross_entropy_solution.py" "$KERNELS/cross_entropy.py"
echo "      参考答案已就位。"

# ─── 运行测试 ─────────────────────────────────────────────────────────────────
echo "[3/4] 运行测试套件..."
echo

# 用 || 捕获失败退出码，最后恢复骨架再报告
TEST_RESULT=0
python3 -m pytest tests/test_kernels.py -v --tb=short 2>&1 || TEST_RESULT=$?

echo

# ─── 恢复骨架 ─────────────────────────────────────────────────────────────────
echo "[4/4] 恢复学生骨架..."
cp "$BACKUP/rmsnorm.py"       "$KERNELS/rmsnorm.py"
cp "$BACKUP/swiglu.py"        "$KERNELS/swiglu.py"
cp "$BACKUP/cross_entropy.py" "$KERNELS/cross_entropy.py"
rm -rf "$BACKUP"
echo "      骨架已恢复，可以开始闯关。"
echo

# ─── 报告结果 ─────────────────────────────────────────────────────────────────
if [ "$TEST_RESULT" -eq 0 ]; then
    echo "✅  全部测试通过！出题包质量 OK。"
    echo "    现在可以开始实现 kernels/{rmsnorm,swiglu,cross_entropy}.py 了。"
    echo "    运行测试：cd m2-kernels && python3 -m pytest tests/test_kernels.py -x -q"
else
    echo "❌  部分测试失败（退出码 $TEST_RESULT）。"
    echo "    这是出题人的 bug，请报告给出题人修复后再开始闯关。"
    echo "    骨架已恢复，参考答案不在工作目录中。"
    exit "$TEST_RESULT"
fi
