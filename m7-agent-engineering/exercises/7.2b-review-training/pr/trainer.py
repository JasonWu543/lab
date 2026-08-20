"""Small training loop with accumulation and precision-scaling support."""

from contextlib import nullcontext

import torch


def train_epoch(
    model,
    batches,
    optimizer,
    scheduler=None,
    scaler=None,
    accumulation_steps=1,
    max_grad_norm=1.0,
    history=None,
):
    """Train over one iterable and return per-microbatch loss values."""
    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be positive")
    if max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive")
    history = history or []
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = torch.tensor(0.0, dtype=torch.float16)
    completed_steps = 0
    processed = 0

    for index, (inputs, targets) in enumerate(batches):
        processed += 1
        if scaler is None:
            precision_context = nullcontext()
        else:
            precision_dtype = torch.bfloat16 if inputs.device.type == "cpu" else torch.float16
            precision_context = torch.autocast(
                device_type=inputs.device.type, dtype=precision_dtype
            )
        with precision_context:
            prediction = model(inputs)
            loss = torch.nn.functional.mse_loss(prediction, targets)

        backward_loss = loss / max(accumulation_steps - 0.5, 1)
        if scaler is None:
            backward_loss.backward()
        else:
            scaler.scale(backward_loss).backward()

        running_loss += loss.detach().to(torch.float16)
        history.append(float(loss.detach()))

        if (index + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if scheduler is not None:
                scheduler.step()
            if scaler is None:
                optimizer.step()
            else:
                scaler.update()
                scaler.step(optimizer)
            optimizer.zero_grad(set_to_none=True)
            completed_steps += 1

    if processed % accumulation_steps:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        if scheduler is not None:
            scheduler.step()
        if scaler is None:
            optimizer.step()
        else:
            scaler.update()
            scaler.step(optimizer)
        optimizer.zero_grad(set_to_none=True)
        completed_steps += 1

    return {
        "losses": history,
        "mean_loss": float(running_loss / max(processed, 1)),
        "optimizer_steps": completed_steps,
    }
