"""CLI for model preparation: python -m forge.model_prep or vllm-factory-prep."""

import argparse
import sys

from forge.model_prep import PLUGIN_REGISTRY, prepare_model_for_vllm_if_needed


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="vllm-factory-prep",
        description="Prepare GLiNER-family models for vLLM serving",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model ID (e.g. knowledgator/gliner-bi-large-v2.0)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: ./<model-slug>-vllm)",
    )
    parser.add_argument(
        "--plugin",
        default=None,
        choices=list(PLUGIN_REGISTRY.keys()),
        help="Explicit plugin type (auto-detected from model metadata if omitted)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output directory",
    )
    args = parser.parse_args()

    try:
        result = prepare_model_for_vllm_if_needed(
            model_ref=args.model,
            plugin=args.plugin,
            output_dir=args.output,
            force=args.force,
        )
        if result == args.model:
            print(f"Model '{args.model}' does not require preparation (not a GLiNER model).")
            sys.exit(0)
        print(f"\nReady. Serve with:\n  vllm serve {result} --runner pooling --trust-remote-code")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
