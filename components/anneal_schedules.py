import math
from typing import Optional


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _progress(t_env: Optional[float], start_t: float, full_t: float) -> float:
    if t_env is None:
        return 1.0
    if full_t <= start_t:
        return 1.0
    return _clamp01((float(t_env) - float(start_t)) / (float(full_t) - float(start_t)))


def _shape_progress(p: float, schedule: str, *, power: float = 2.0, k: float = 10.0, midpoint: float = 0.5) -> float:
    p = _clamp01(p)
    schedule = (schedule or "linear").lower()

    if schedule == "linear":
        return p

    if schedule == "cosine":
        return 0.5 * (1.0 - math.cos(math.pi * p))

    if schedule == "sqrt":
        return math.sqrt(p)

    if schedule == "square":
        return p * p

    if schedule in {"poly", "power"}:
        if power <= 0:
            return p
        return p ** float(power)

    if schedule == "exp":
        kk = float(k)
        if abs(kk) < 1e-12:
            return p
        # Normalize so f(0)=0, f(1)=1
        num = math.exp(kk * p) - 1.0
        den = math.exp(kk) - 1.0
        if abs(den) < 1e-12:
            return p
        return _clamp01(num / den)

    if schedule == "sigmoid":
        kk = float(k)
        mid = float(midpoint)

        def sig(x: float) -> float:
            return 1.0 / (1.0 + math.exp(-kk * (x - mid)))

        s0 = sig(0.0)
        s1 = sig(1.0)
        sp = sig(p)
        den = (s1 - s0)
        if abs(den) < 1e-12:
            return p
        return _clamp01((sp - s0) / den)

    raise ValueError(
        f"Unknown mask_backward_schedule={schedule!r}. Expected one of: linear, cosine, sqrt, square, poly, exp, sigmoid."
    )


def mask_backward_prob_from_args(args, t_env: Optional[float]) -> float:
    """Compute probability of masking the backward (future) window.

    Uses args:
      - mask_backward_schedule: linear|cosine|sqrt|square|poly|exp|sigmoid
      - mask_backward_start_t_env
      - mask_backward_full_t_env
      - mask_backward_start
      - mask_backward_finish
      - mask_backward_power (for poly)
      - mask_backward_k (for exp/sigmoid)
      - mask_backward_midpoint (for sigmoid)

    Fallback (if *_full_t_env is missing): full masking at 0.8 * args.t_max.

    If args.causal_window is True: returns 1.0.
    """

    if getattr(args, "causal_window", False):
        return 1.0

    schedule = getattr(args, "mask_backward_schedule", "linear")
    start_t = float(getattr(args, "mask_backward_start_t_env", 0.0))

    full_t_env = getattr(args, "mask_backward_full_t_env", None)
    if full_t_env is None:
        t_max = getattr(args, "t_max", None)
        if t_max is None:
            full_t = start_t
        else:
            full_t = 0.8 * float(t_max)
    else:
        full_t = float(full_t_env)

    start = float(getattr(args, "mask_backward_start", 0.0))
    finish = float(getattr(args, "mask_backward_finish", 1.0))

    power = float(getattr(args, "mask_backward_power", 2.0))
    k = float(getattr(args, "mask_backward_k", 10.0))
    midpoint = float(getattr(args, "mask_backward_midpoint", 0.5))

    p = _progress(t_env, start_t, full_t)
    shaped = _shape_progress(p, schedule, power=power, k=k, midpoint=midpoint)
    value = start + (finish - start) * shaped

    # Masking probability should be a probability.
    return _clamp01(float(value))
