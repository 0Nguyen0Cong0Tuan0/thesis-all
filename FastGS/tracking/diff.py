import json
from argparse import Namespace
from typing import Any, Dict


def to_plain(obj: Any) -> Any:
    """Convert objects to JSON-serializable python types."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [to_plain(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, Namespace):
        return {k: to_plain(v) for k, v in vars(obj).items()}

    # Torch / numpy objects are handled best-effort without importing them here.
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            return obj.item()
        except Exception:
            pass

    if hasattr(obj, "tolist") and callable(getattr(obj, "tolist")):
        try:
            return obj.tolist()
        except Exception:
            pass

    return str(obj)


def diff_dict(prev: Dict[str, Any], curr: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow diff: keys whose values changed (JSON-serializable)."""
    out: Dict[str, Any] = {}
    keys = set(prev.keys()) | set(curr.keys())
    for k in keys:
        if prev.get(k) != curr.get(k):
            out[k] = curr.get(k)
    return out


def json_dumps(obj: Any) -> str:
    return json.dumps(to_plain(obj), ensure_ascii=False, sort_keys=True)
