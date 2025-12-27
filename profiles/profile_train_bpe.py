import argparse
import json
import base64
import cProfile
import pstats
from io import StringIO

from cs336_basics.tokenizer import train_bpe


def b64_encode(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def main():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer.")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument(
        "--tokenizer_output_path",
        type=str,
        default="tokenizer.json",
        help="Path to save the tokenizer JSON file.",
    )
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    args = parser.parse_args()

    special_tokens = ["<|endoftext|>"]

    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()

    # ---- Train BPE ----
    vocab, merges = train_bpe(
        args.input_path,
        args.vocab_size,
        special_tokens=special_tokens,
    )
    
    if args.profile:
        profiler.disable()
        
        # Print to console
        s = StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(30)  # Top 30 functions
        print(s.getvalue())
        
        # Save to file
        profiler.dump_stats('bpe_training.prof')
        print("Profile saved to bpe_training.prof")

    # ---- Serialize ----
    tokenizer_json = {
        "version": "cs336-bpe-v1",
        "type": "bpe",
        "byte_encoding": "base64",
        "special_tokens": special_tokens,
        "vocab": [],
        "merges": [],
    }

    # vocab: dict[int, bytes] -> list[{id, token}]
    for token_id in sorted(vocab.keys()):
        token_bytes = vocab[token_id]
        tokenizer_json["vocab"].append(
            {
                "id": token_id,
                "token": b64_encode(token_bytes),
            }
        )

    # merges: list[tuple[bytes, bytes]]
    for a, b in merges:
        tokenizer_json["merges"].append(
            [b64_encode(a), b64_encode(b)]
        )

    # ---- Write file ----
    with open(args.tokenizer_output_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_json, f, indent=2)

    print(f"Tokenizer saved to {args.tokenizer_output_path}")


if __name__ == "__main__":
    main()
