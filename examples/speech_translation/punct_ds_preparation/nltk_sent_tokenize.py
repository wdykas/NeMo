import argparse
from pathlib import Path

import nltk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", "-i", type=Path, required=True)
    parser.add_argument("--output_file", "-o", type=Path, required=True)
    args = parser.parse_args()
    args.input_file = args.input_file.expanduser()
    args.output_file = args.output_file.expanduser()
    return args


def main() -> None:
    args = parse_args()
    with args.input_file.open() as in_f, args.output_file.open('w') as out_f:
        for line in in_f:
            for sent in nltk.sent_tokenize(line):
                out_f.write(sent.rstrip() + '\n')


if __name__ == "__main__":
    main()
