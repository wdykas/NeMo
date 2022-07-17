import argparse
import re
from pathlib import Path

from tqdm import tqdm


MARKUP = re.compile('^</doc>.*\n|^<doc docid=.*\n', flags=re.MULTILINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--inputs", nargs="+", type=Path)
    input_group.add_argument("--input_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.expanduser()
    if args.inputs is None:
        args.input_dir = args.input_dir.expanduser()
    else:
        args.inputs = [elem.expanduser() for elem in args.inputs]
    return args


def main() -> None:
    args = parse_args()
    input_files = list(args.input_dir.iterdir()) if args.inputs is None else args.inputs
    with args.output.open('w') as out_f:
        for input_file in tqdm(input_files, desc="Uniting documents", unit="file"):
            with input_file.open() as in_f:
                out_f.write(MARKUP.sub('', in_f.read()))


if __name__ == "__main__":
    main()
