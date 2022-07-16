import argparse
import os
import random
from pathlib import Path
from subprocess import PIPE, run
from typing import Union

from tqdm import tqdm

import prepare_small_data_for_punctuation_capitalization_task as small


BUFF_SIZE = 2 ** 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--start_length", type=int, default=3)
    parser.add_argument("--end_length", type=int, default=128)
    parser.add_argument("--num_passes_through_dataset", type=int, default=1)
    args = parser.parse_args()
    args.input_file = args.input_file.expanduser()
    args.output_file = args.output_file.expanduser()
    return args


def get_num_characters(input_file: Union[str, os.PathLike]) -> int:
    result = run(['wc', '-m', str(input_file)], stdout=PIPE, stderr=PIPE)
    if not result:
        raise ValueError(
            f"Bash command `wc -m {input_file}` returned and empty string. "
            f"Possibly, file {input_file} does not exist."
        )
    return int(result.stdout.decode('utf-8').split()[0])


def main() -> None:
    args = parse_args()
    perm = list(range(args.start_length, args.end_length))
    random.shuffle(perm)
    p_i = 0  # index of segment length in permutation `perm`
    b_i = 0  #
    with args.output_file.open('w', buffering=BUFF_SIZE) as out_f:
        for _ in range(args.num_passes_throuh_dataset):
            with args.input_file.open(buffering=BUFF_SIZE) as in_f:
                buff = in_f.read(BUFF_SIZE).replace('\n', ' ')
                while True:
                    if p_i == len(perm):
                        p_i = 0
                        random.shuffle(perm)
                    if args.end_length >= len(small.WORD_WITH_PRECEDING_AND_FOLLOWING_PUNCTUATION.findall(buff[b_i:])):
                        buff = buff[b_i:]
                        b_i = 0
                        buff += in_f.read(BUFF_SIZE).replace('\n', ' ')
                    found_required_length = False
                    for m_i, m in enumerate(small.WORD_WITH_PRECEDING_AND_FOLLOWING_PUNCTUATION.finditer(buff[b_i:])):
                        if m_i >= perm[p_i] - 1:
                            out_f.write(buff[b_i : b_i + m.span()[1]] + '\n')
                            b_i = b_i + m.span()[1]
                            found_required_length = True
                            break
                    if not found_required_length:
                        out_f.write(buff[b_i :])
                    p_i += 1


if __name__ == "__main__":
    main()
