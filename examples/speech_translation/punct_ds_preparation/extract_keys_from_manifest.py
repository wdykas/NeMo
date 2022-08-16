import argparse
import json

from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", required=True, type=Path)
    parser.add_argument("--output", "-o", required=True, type=Path)
    parser.add_argument("--key", required=True)
    args = parser.parse_args()
    args.input = args.input.expanduser()
    args.output = args.output.expanduser()
    return args


def main() -> None:
    args = parse_args()
    with args.input.open() as in_f:
        lines = in_f.readlines()
    extracted = [json.loads(line)[args.key].strip() for line in lines]
    args.output.parent.mkdir(exist_ok=True, parents=True)
    with args.output.open('w') as out_f:
        for line in extracted:
            out_f.write(line + '\n')


if __name__ == "__main__":
    main()