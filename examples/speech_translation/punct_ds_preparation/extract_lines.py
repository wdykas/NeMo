import argparse
import os
import random
from pathlib import Path
from subprocess import PIPE, run
from typing import Union

from tqdm import tqdm


BUFFER_SIZE = 2 ** 25
MIN_FRACTION_OF_NUMBER_OF_EXTRACTED_LINES_FOR_SAMPLING = 0.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--extracted_file", type=Path, required=True)
    parser.add_argument("--remaining_file", type=Path, required=True)
    parser.add_argument(
        "--num_lines_to_extract",
        type=int,
        default=5000,
    )
    args = parser.parse_args()
    for name in ["input_file", "extracted_file", "remaining_file"]:
        setattr(parser, name, getattr(args, name).expanduser())
    return args


def get_num_lines(input_file: Union[str, os.PathLike]) -> int:
    result = run(['wc', '-l', str(input_file)], stdout=PIPE, stderr=PIPE)
    if not result:
        raise ValueError(
            f"Bash command `wc -l {input_file}` returned and empty string. "
            f"Possibly, file {input_file} does not exist."
        )
    return int(result.stdout.decode('utf-8').split()[0])


def main() -> None:
    args = parse_args()
    num_lines = get_num_lines(args.input_file)
    if args.num_lines_to_extract / num_lines < MIN_FRACTION_OF_NUMBER_OF_EXTRACTED_LINES_FOR_SAMPLING:
        extracted_line_indices = {random.randrange(0, num_lines) for _ in range(args.num_lines_to_extract)}
        while len(extracted_line_indices) < args.num_lines_to_extract:
            extracted_line_indices.add(random.randrange(0, num_lines))
    else:
        all_indices = list(range(num_lines))
        extracted_line_indices = set(random.sample(all_indices, args.num_lines_to_extract))
    with args.input_file.open(buffering=BUFFER_SIZE) as in_f, \
            args.extracted_file.open('w', buffering=BUFFER_SIZE) as e_f, \
            args.remaining_file.open('w', buffering=BUFFER_SIZE) as r_f:
        for i, line in enumerate(tqdm(in_f, unit="line", total=num_lines, desc="Extracting lines")):
            if i in extracted_line_indices:
                e_f.write(line)
            else:
                r_f.write(line)


if __name__ == "__main__":
    main()