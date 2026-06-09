"""Guard test: nothing in ml/, storage/, normalization/, schemas/, api/ imports from serving/."""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "ticket_price_predictor"


def test_no_upward_imports() -> None:
    """ml/, storage/, normalization/, schemas/, api/ must not import from serving/."""
    forbidden = re.compile(
        r"from\s+ticket_price_predictor\.serving"  # from ticket_price_predictor.serving …
        r"|import\s+ticket_price_predictor\.serving"  # import ticket_price_predictor.serving (as …)
        r"|from\s+\.+serving"  # from .serving / from ..serving
        r"|from\s+\.+\s+import\s+(?:[\w,\s]*,\s*)?serving"  # from . import serving / from .. import a, serving
        r"|from\s+ticket_price_predictor\s+import\s+(?:[\w,\s]*,\s*)?serving"  # from ticket_price_predictor import serving
    )
    for sub in ["ml", "storage", "normalization", "schemas", "api"]:
        for path in (SRC / sub).rglob("*.py"):
            assert not forbidden.search(path.read_text()), f"{path} imports from serving/"
