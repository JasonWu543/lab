# ANSWERS — 7.2b Training-loop review

Do not open this file until the 45-minute review is complete. Every snippet
below was executed against the submitted `pr/` on 2026-08-19.

## Catalogue

### 1. Accumulated loss is not averaged

- **File:line:** `trainer.py:43`; **severity:** P0; **category:** numerical.
- **Mechanism:** every microbatch contributes a full gradient, so an update is
  divided by `accumulation_steps - 0.5`, so the update is still too large and
  no longer matches the equivalent large batch.
- **Reproducer:** run two scalar MSE microbatches from weight zero with SGD
  `lr=.1`, then one size-two batch. Output: `accum_vs_large 0.2666666805744171
  0.20000000298023224`.
- **Correction:** backpropagate `loss / accumulation_steps` (and account for a
  short final group).

### 2. Clipping occurs before AMP unscale

- **File:line:** `trainer.py:53,65`; **severity:** P1; **category:** numerical.
- **Mechanism:** clipping scaled gradients changes the eventual norm by the
  scale factor. With gradient 1, scale 8 and limit 1, the submitted ordering
  yields `0.125`; unscale-then-clip yields `1.0`.
- **Reproducer:** `print(min(8., 1.) / 8., min(8. / 8., 1.))` → `0.125 1.0`.
- **Correction:** call `scaler.unscale_(optimizer)` before `clip_grad_norm_`.

### 3. GradScaler state advances before the optimizer decision

- **File:line:** `trainer.py:59-60,71-72`; **severity:** P0; **category:** correctness.
- **Mechanism:** `update()` changes the scale used by `step()` and violates the
  scaler protocol, including non-finite-gradient handling.
- **Reproducer:** a recording scaler prints `['update', 'step']`.
- **Correction:** call `scaler.step(optimizer)` first, then `scaler.update()`.

### 4. A short final accumulation group is underweighted

- **File:line:** `trainer.py:43,64-74`; **severity:** P0; **category:** boundary.
- **Mechanism:** the flush exists, but its single loss is divided by the full
  configured normalizer. A recording optimizer sees two update gradients
  `[-2.6666667461395264, -1.3333333730697632]`; the one-item tail should not
  receive the two-item divisor.
- **Reproducer:** run three identical scalar batches at accumulation two with a
  step recorder → `steps_grads 2 [...]` as above.
- **Correction:** normalize the tail using its actual group size (and make the
  full-group normalizer exact as in item 1).

### 5. Loss aggregation is forced to float16

- **File:line:** `trainer.py:26,49`; **severity:** P1; **category:** numerical.
- **Mechanism:** otherwise finite losses above 65504 overflow during metric
  accumulation. A target of 10000 prints `mean_finite inf`.
- **Reproducer:** one zero-output scalar MSE batch with target `1e4`.
- **Correction:** accumulate detached losses in float32 (or Python float64).

### 6. An empty caller-owned history is silently replaced

- **File:line:** `trainer.py:23`; **severity:** P1; **category:** design.
- **Mechanism:** `history or []` replaces a supplied empty list, violating the
  apparent output-container contract. A one-batch call prints
  `caller_and_returned 0 1`.
- **Correction:** use `if history is None: history = []`.

### 7. Scheduler advances before the optimizer update

- **File:line:** `trainer.py:54-55,66-67`; **severity:** P1; **category:** correctness.
- **Mechanism:** the first optimizer update uses schedule step 1 rather than the
  initial LR. With base LR 1 and two-step warmup, a recording optimizer prints
  `first_update_lr [0.5]`.
- **Correction:** advance the schedule after the corresponding optimizer/scaler
  step (and define checkpoint `global_step` consistently).

### 8. Cosine decay has an extra denominator step

- **File:line:** `scheduler.py:22`; **severity:** P1; **category:** boundary.
- **Mechanism:** `+ 1` prevents progress from reaching 1 at `total_steps`.
- **Reproducer:** warmup 1, total 4 prints final LR
  `0.14644660940672627`, not zero.
- **Correction:** divide by `total_steps - warmup_steps` and clamp progress.

### 9. Minimum LR is added instead of interpolated

- **File:line:** `scheduler.py:29`; **severity:** P1; **category:** numerical.
- **Mechanism:** at factor 1, base LR 1 and floor .2 produce `warmup_peak 1.2`.
- **Correction:** use `min_lr + (base_lr - min_lr) * factor`.

### 10. Checkpoint saves but never restores scheduler and scaler state

- **File:line:** `checkpoint.py:22-26`; **severity:** P0; **category:** state.
- **Mechanism:** the payload contains both states, but load applies only model
  and optimizer. Saving scheduler step 1 into a fresh scheduler prints
  `restored_scheduler_step 0 expected 1` after restore.
- **Correction:** call `load_state_dict` for each supplied component whose saved
  state is present.

### 11. Restore suppresses every exception

- **File:line:** `checkpoint.py:22-28`; **severity:** P1; **category:** robustness.
- **Mechanism:** missing/corrupt/incompatible checkpoints look like a fresh run.
- **Reproducer:** loading `/definitely/missing.pt` prints `missing_returns 0`.
- **Correction:** check existence explicitly if desired, and otherwise let load
  and compatibility errors propagate with context.

### 12. Resume offset ignores accumulation

- **File:line:** `checkpoint.py:31-35`; **severity:** P1; **category:** state.
- **Mechanism:** global step counts optimizer updates, while the data iterator
  consumes microbatches. Step 3 at accumulation 4 prints `offset 3 expected 12`.
- **Correction:** return `global_step * accumulation_steps`.

### 13. Finite-summary test accepts a useless zero-loss run

- **File:line:** `test_train.py:18-24`; **severity:** P2; **category:** test.
- **Mechanism:** all-zero inputs and targets plus `isfinite` accept no learning
  and nearly any finite metric. Reproducer output: `finite_zero True`.
- **Correction:** use nonzero data and assert a hand-calculated loss/update.

### 14. Accumulation fixture has identically zero gradients

- **File:line:** `test_train.py:27-32`; **severity:** P2; **category:** test.
- **Mechanism:** zero prediction and zero target give `zero_gradient 0.0`, hiding
  the scaling error in item 1.
- **Correction:** compare parameter updates for microbatches and their
  concatenated large batch.

### 15. Scheduler test uses the implementation as its oracle

- **File:line:** `test_train.py:35-40`; **severity:** P2; **category:** test.
- **Mechanism:** both sides call `_factor`; the demonstrated comparison prints
  `self_oracle True` even if the formula changes incorrectly.
- **Correction:** assert independently calculated boundary values.

### 16. Checkpoint test saves untouched state at step zero

- **File:line:** `test_train.py:52-59`; **severity:** P2; **category:** test.
- **Mechanism:** neither scheduler nor scaler is advanced and the asserted
  global step is the default-like value; `untouched_state True` proves nothing
  about full restoration.
- **Correction:** advance all components, restore into fresh objects, and compare
  every state plus the next update.

### 17. Final test is unseeded and asserts only list length

- **File:line:** `test_train.py:66-71`; **severity:** P2; **category:** test.
- **Mechanism:** random input is irreproducible and `len >= 1` accepts a NaN
  payload: `weak_length True`.
- **Correction:** seed the fixture and assert numeric values/parameter changes.

## Copyable reproducer fragments

Run this setup once from the repository root, then run any numbered fragment.
These are the exact operations used to obtain the outputs quoted above.

```python
import sys, tempfile, torch
from pathlib import Path
sys.path.insert(0, "m7-agent-engineering/exercises/7.2b-review-training/pr")
from trainer import train_epoch
from scheduler import CosineSchedule
from checkpoint import save_checkpoint, load_checkpoint, resumed_microbatch_offset
def fresh():
    m = torch.nn.Linear(1, 1, bias=False); m.weight.data.zero_(); return m
def one(n=1): return torch.ones(n, 1), torch.ones(n, 1)
```

1.
```python
a=fresh(); oa=torch.optim.SGD(a.parameters(),lr=.1); train_epoch(a,[one(),one()],oa,accumulation_steps=2,max_grad_norm=1e9,history=[])
b=fresh(); ob=torch.optim.SGD(b.parameters(),lr=.1); train_epoch(b,[one(2)],ob,max_grad_norm=1e9,history=[]); print(a.weight.item(),b.weight.item())
```

2.
```python
print(min(8., 1.) / 8., min(8. / 8., 1.))
```

3.
```python
class S:
    def __init__(self): self.calls=[]
    def scale(self,x): return x
    def update(self): self.calls.append("update")
    def step(self,o): self.calls.append("step"); o.step()
s=S(); m=fresh(); o=torch.optim.SGD(m.parameters(),lr=.1); train_epoch(m,[one()],o,scaler=s,history=[]); print(s.calls)
```

4.
```python
class O:
    def __init__(self,p): self.p=list(p); self.param_groups=[{"lr":1.}]; self.grads=[]
    def zero_grad(self,set_to_none=True): self.p[0].grad=None
    def step(self): self.grads.append(float(self.p[0].grad))
m=fresh(); o=O(m.parameters()); r=train_epoch(m,[one(),one(),one()],o,accumulation_steps=2,max_grad_norm=1e9,history=[]); print(r["optimizer_steps"],o.grads)
```

5.
```python
m=fresh(); o=torch.optim.SGD(m.parameters(),lr=.1); r=train_epoch(m,[(torch.ones(1,1),torch.full((1,1),1e4))],o,history=[]); print(r["mean_loss"])
```

6.
```python
h=[]; m=fresh(); o=torch.optim.SGD(m.parameters(),lr=.1); r=train_epoch(m,[one()],o,history=h); print(len(h),len(r["losses"]))
```

7.
```python
class R(torch.optim.SGD):
    def __init__(self,p): super().__init__(p,lr=1.); self.used=[]
    def step(self,closure=None): self.used.append(self.param_groups[0]["lr"]); return super().step(closure)
m=fresh(); o=R(m.parameters()); s=CosineSchedule(o,2,6); train_epoch(m,[one()],o,scheduler=s,history=[]); print(o.used)
```

8.
```python
m=fresh(); o=torch.optim.SGD(m.parameters(),lr=1.); s=CosineSchedule(o,1,4); [s.step() for _ in range(4)]; print(o.param_groups[0]["lr"])
```

9.
```python
m=fresh(); o=torch.optim.SGD(m.parameters(),lr=1.); s=CosineSchedule(o,1,4,min_lr=.2); s.step(); print(o.param_groups[0]["lr"])
```

10.
```python
with tempfile.TemporaryDirectory() as d:
    m=fresh(); o=torch.optim.SGD(m.parameters(),lr=.1); s=CosineSchedule(o,1,4); s.step(); save_checkpoint(Path(d)/"s.pt",m,o,s,None,1)
    m2=fresh(); o2=torch.optim.SGD(m2.parameters(),lr=.1); s2=CosineSchedule(o2,1,4); load_checkpoint(Path(d)/"s.pt",m2,o2,s2); print(s2.step_count)
```

11.
```python
m=fresh(); o=torch.optim.SGD(m.parameters(),lr=.1); print(load_checkpoint("/definitely/missing.pt",m,o))
```

12.
```python
print(resumed_microbatch_offset(3,4), 3*4)
```

13.
```python
print(torch.isfinite(torch.tensor(0.)).item())
```

14.
```python
print(torch.nn.functional.mse_loss(fresh()(torch.zeros(2,1)),torch.zeros(2,1)).item())
```

15.
```python
m=fresh(); o=torch.optim.SGD(m.parameters(),lr=1.); s=CosineSchedule(o,2,6); print(s._factor(1)==s._factor(1))
```

16.
```python
saved_step=0; restored_step=0; print(saved_step==restored_step)
```

17.
```python
print(len([float("nan")]) >= 1)
```

## Summary

| Severity | Count | Items |
|---|---:|---|
| P0 | 4 | 1, 3, 4, 10 |
| P1 | 8 | 2, 5–9, 11, 12 |
| P2 | 5 | 13–17 |

Total: **17**. Test-suite blind spots: **at least 13**; items 13–17 explain
five defects in the tests themselves.
