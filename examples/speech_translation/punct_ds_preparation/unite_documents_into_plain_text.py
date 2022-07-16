import argparse
import re
from pathlib import Path

from tqdm import tqdm


MARKUP = re.compile('^<doc/>.*\n|^<doc docid=.*\n', flags=re.MULTILINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.expanduser()
    args.inputs = [elem.expanduser() for elem in args.inputs]
    return args


def main() -> None:
    args = parse_args()
    with args.output.open('w') as out_f:
        for input_file in tqdm(args.inputs, desc="Uniting documents", unit="file"):
            with input_file.open() as in_f:
                out_f.write(MARKUP.sub('', in_f.read()))


if __name__ == "__main__":
    main()
