import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument(
        "--borders", nargs="+", type=int, help="Indices of first lines in blocks starting with 0.", required=True
    )
    args = parser.parse_args()
    parser.input_file = parser.input_file.expanduser()
    parser.output_file = parser.output_file.expanduser()
    return args


def main() -> None:
    args = parse_args()
    borders = sorted(args.borders)
    if borders[0] == 0:
        borders = borders[1:]
    current_block_idx = 0
    current_block = ""
    with args.output_file.open('w') as out_f, args.input_file.open() as in_f:
        for i, line in enumerate(in_f):
            if i >= borders[current_block_idx]:
                out_f.write(current_block + '\n')
                current_block = ""
                current_block_idx += 1
            line = line.strip()
            if current_block:
                current_block += ' '
            current_block += line
        out_f.write(current_block + '\n')


if __name__ == "__main__":
    main()