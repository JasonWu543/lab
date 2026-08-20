"""扫描 arrival rate × chunk size，输出 PD 延迟分位数表。"""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minipd.simulate import make_workload, run_sim
from minipd.transfer import TransferModel


def main() -> None:
    print("arrival_rate chunk_size p50_ttft p95_ttft p50_tpot p95_tpot")
    for rate in (0.25, 0.5, 1.0):
        for chunk in (4, 16, 64):
            workload = make_workload(100, 4200, rate, (8, 256), (4, 32))
            result = run_sim(workload, 64, chunk, 8, TransferModel(1, 16, 1024))
            ttft = np.fromiter(result.ttft.values(), dtype=float)
            tpot = np.fromiter(result.tpot.values(), dtype=float)
            print(f"{rate:12.2f} {chunk:10d} {np.percentile(ttft, 50):8.2f} "
                  f"{np.percentile(ttft, 95):8.2f} {np.percentile(tpot, 50):8.2f} "
                  f"{np.percentile(tpot, 95):8.2f}")


if __name__ == "__main__":
    main()
