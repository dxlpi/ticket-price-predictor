#!/usr/bin/env python3
"""Run the prediction web app."""

import argparse
import os
import sys
from pathlib import Path

import uvicorn

from ticket_price_predictor.serving.app import create_app
from ticket_price_predictor.serving.dependencies import init_dependencies

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = REPO_ROOT / "web"
DATA_DIR = REPO_ROOT / "data"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    model_path = args.model_path or (
        Path(os.environ["MODEL_PATH"]) if "MODEL_PATH" in os.environ else None
    )
    if model_path is None:
        print(
            "ERROR: model path not set (use --model-path or MODEL_PATH env var)",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        init_dependencies(model_path=model_path, data_dir=DATA_DIR)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    app = create_app(static_dir=WEB_DIR)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
