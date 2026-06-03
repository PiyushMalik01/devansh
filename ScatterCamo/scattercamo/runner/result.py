"""Unified attack result so every attack plugs into the same runner + metrics."""

from dataclasses import dataclass, field


@dataclass
class AttackResult:
    adv_image: object               # np.ndarray (h, w, 3) or None if no adversarial found
    success: bool
    queries: int
    history: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)


def normalize_result(result):
    """Coerce any attack's output into an ``AttackResult``.

    Accepts an ``AttackResult`` (returned as-is) or a dict with ``success`` and
    ``queries`` plus either ``adv_image`` or ``best`` (a Solution exposing
    ``generate_image()``) -- the shape ScatterCamo returns.
    """
    if isinstance(result, AttackResult):
        return result
    if isinstance(result, dict):
        adv = result.get("adv_image")
        if adv is None and result.get("best") is not None:
            adv = result["best"].generate_image()
        return AttackResult(
            adv_image=adv,
            success=bool(result.get("success", adv is not None)),
            queries=int(result.get("queries", 0)),
            history=result.get("history", []),
            extra={k: v for k, v in result.items()
                   if k not in ("adv_image", "best", "success", "queries", "history")},
        )
    raise TypeError(f"Cannot normalize attack result of type {type(result)}")
