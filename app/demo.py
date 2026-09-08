"""Command-line entry point for the Phase 7 Gradio demo."""

from __future__ import annotations

from argparse import ArgumentParser

from demo.interface import build_interface


def main() -> None:
    parser = ArgumentParser(description="Launch super-resolution demo.")
    parser.add_argument("--share", action="store_true", help="Public Colab link.")
    parser.add_argument("--server-name", default=None)
    parser.add_argument("--server-port", type=int, default=None)
    args = parser.parse_args()
    build_interface().launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
    )


if __name__ == "__main__":
    main()
