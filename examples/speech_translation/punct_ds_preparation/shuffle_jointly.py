import argparse
import logging
import os
import random
import shutil
from pathlib import Path
from subprocess import run, PIPE

from tqdm import tqdm


logging.basicConfig(level="INFO", format='%(levelname)s -%(asctime)s - %(name)s - %(message)s')

BUFFER_SIZE = 2 ** 25


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--input_files",
        type=Path,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--output_files",
        type=Path,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--line_delimiter",
        required=True,
        help="It has to be a character which does not occur in any of input files, e.g. tabulation '\\t'."
    )
    parser.add_argument(
        "--tmp_dir",
        type=Path,
        help="Path to a directory where temporary files like `--united_file_name`, `--shuffled_file_name`, "
        "a `--split_dir` directory are created. Be default it is the directory containing "
        "the first input file.",
    )
    parser.add_argument(
        "--united_file_name",
        default="united_lines.txt",
        help="Temporary file where united lines from all input files are stored before shuffling with `shuf` util.",
    )
    parser.add_argument(
        "--shuffled_file_name",
        default="shuffled_lines.txt",
        help="Temporary file where united lines from all input files after shuffling with `shuf` util.",
    )
    parser.add_argument(
        "--max_shuf_lines",
        type=int,
        help="Maximum number of lines in a file which can be shuffled by 1 `shuf` call. If the number of lines in the "
        "united file is greater than `--max_shuf_lines`, then united file is split into smaller files of size "
        "`--num_split_lines` and "
        "then concatenated in random order and saved into a file with name `--united_file_name` located in the "
        "`--tmp-dir`. After that concatenated file is split into parts, parts are shuffled, and "
        "then shuffled parts are concatenated",
        default=6 * 10 ** 7,
    )
    parser.add_argument(
        "--num_split_lines",
        type=int,
        help="Number of lines files which are created for shuffling very large files. See more in `--max_shuf_lines` "
        "parameter description.",
        default=1000,
    )
    parser.add_argument(
        "--split_dir",
        help="A name of a directory where split united file parts are saved. See more in `--max_shuf_lines` parameter "
        "description.",
        default="split_files",
    )
    parser.add_argument(
        "--resume_from",
        choices=[
            'shuffling',
            'uniting_small_files',
            'splitting_for_shuf_util',
            'applying_shuf_to_smaller_files',
            'uniting_smaller_shufed_files',
        ]
    )
    args = parser.parse_args()
    for i, f in enumerate(args.input_files):
        args.input_files[i] = f.expanduser()
    for i, f in enumerate(args.output_files):
        args.output_files[i] = f.expanduser()
    if args.tmp_dir is None:
        args.tmp_dir = args.input_files[0].parent
    else:
        args.tmp_dir = args.tmp_dir.expanduser()
    if len(args.input_files) != len(args.output_files):
        parser.error("Number of elements in parameters `--input_files` and `--output_file` has to be equal")
    return args


def get_num_lines(input_file):
    result = run(['wc', '-l', str(input_file)], stdout=PIPE, stderr=PIPE)
    if not result:
        raise ValueError(
            f"Bash command `wc -l {input_file}` returned and empty string. "
            f"Possibly, file {input_file} does not exist."
        )
    return int(result.stdout.decode('utf-8').split()[0])


def shuffle_with_splitting(
    united_file_path: Path,
    shuffled_file_path: Path,
    max_shuf_lines: int,
    split_dir: str,
    num_split_lines: int,
    num_lines: int,
    resume_from: str,
) -> None:
    split_dir = united_file_path.parent / split_dir
    if resume_from not in [
        'uniting_small_files',
        'splitting_for_shuf_util',
        'applying_shuf_to_smaller_files',
        'uniting_smaller_shufed_files',
    ]:
        logging.info(f"Removing existing files in {split_dir}.")
        if split_dir.exists():
            shutil.rmtree(str(split_dir))
        split_dir.mkdir(exist_ok=True, parents=True)
        logging.info(
            f"Splitting large file {united_file_path} into very small files which will be permuted:\n"
            f"split --lines {num_split_lines} {united_file_path} {split_dir}/x"
        )
        run(["split", "--lines", f"{num_split_lines}", f"{united_file_path}", f"{split_dir}/x"], check=True)
        logging.info(f"Split {united_file_path} successfully.")
    if resume_from not in [
        'splitting_for_shuf_util', 'applying_shuf_to_smaller_files', 'uniting_smaller_shufed_files'
    ]:
        files = list(split_dir.iterdir())
        random.shuffle(files)
        logging.info("Uniting small files")
        with united_file_path.open('w') as out_f:
            for file in files:
                with file.open() as in_f:
                    in_text = in_f.read()
                    out_f.write(in_text + ('' if in_text[-1] == '\n' else '\n'))
    if resume_from not in ['applying_shuf_to_smaller_files', 'uniting_smaller_shufed_files']:
        num_splits = 2
        if max_shuf_lines < 1:
            raise ValueError("Parameter `max_shuf_lines` must be positive.")
        while num_lines // num_splits > max_shuf_lines:
            num_splits += 1
        logging.info(f"Cleaning split dir {split_dir}...")
        shutil.rmtree(str(split_dir))
        split_dir.mkdir()
        logging.info(
            f"Splitting large file {united_file_path} into smaller files for processing with `shuf` util:\n"
            f"split --number l/{num_splits} {united_file_path} {split_dir}/x"
        )
        run(["split", "--number", f"l/{num_splits}", f"{united_file_path}", f"{split_dir}/x"], check=True)
        logging.info(f"Split {united_file_path} successfully.")
    if resume_from not in ['uniting_smaller_shufed_files']:
        shuffled_files = []
        for file in split_dir.iterdir():
            shuffled_file = Path(str(file) + '.shuf')
            shuffled_files.append(shuffled_file)
            logging.info(f"Applying `shuf` util to file {file}:\nshuf {file} > {shuffled_file}")
            with shuffled_file.open('w') as f:
                run(['shuf', str(file)], stdout=f, check=True)
            logging.info(f"Successfully applied `shuf` to {file}")
            file.unlink()
    with shuffled_file_path.open('w') as f:
        logging.info(
            f"Uniting shuffled files:\n"
            f"cat {' '.join([str(file) for file in split_dir.iterdir() if file.suffix == '.shuf'])} > "
            f"{shuffled_file_path}"
        )
        run(['cat'] + [str(file) for file in split_dir.iterdir() if file.suffix == '.shuf'], stdout=f, check=True)
        logging.info("Successfully united files.")
    shutil.rmtree(str(split_dir))


def main():
    args = get_args()
    input_file_objects = [inp_file.open(buffering=BUFFER_SIZE) for inp_file in args.input_files]
    united_file_path = args.tmp_dir / args.united_file_name
    args.tmp_dir.mkdir(parents=True, exist_ok=True)
    lines = [inp_obj.readline().strip('\n') for inp_obj in input_file_objects]
    line_number = 0
    num_lines = get_num_lines(args.input_files[0])
    if args.resume_from is None:
        progress_bar = tqdm(total=num_lines, unit='line', desc="Uniting files", unit_scale=True)
        with united_file_path.open('w', buffering=BUFFER_SIZE) as united_f:
            while all(lines):
                delimiter_in_line = [args.line_delimiter in line for line in lines]
                if any(delimiter_in_line):
                    raise ValueError(
                        f"Line delimiter {repr(args.line_delimiter)} is present in line number {line_number} in file "
                        f"{args.input_files[delimiter_in_line.index(True)]}."
                    )
                united_f.write(args.line_delimiter.join(lines) + '\n')
                progress_bar.n += 1
                progress_bar.update(0)
                lines = [inp_obj.readline().strip('\n') for inp_obj in input_file_objects]
        progress_bar.close()
        if any(lines):
            raise ValueError(
                f"Files {', '.join([str(args.input_files[i]) for i, line in enumerate(lines) if not line])} have less "
                f"lines than files {', '.join([str(args.input_files[i]) for i, line in enumerate(lines) if line])}."
            )
        for input_file_object in input_file_objects:
            input_file_object.close()
    shuffled_file_path = args.tmp_dir / args.shuffled_file_name
    if num_lines < args.max_shuf_lines:
        logging.info(f"Shuffling: shuf {united_file_path} > {shuffled_file_path}")
        with shuffled_file_path.open('w') as f:
            run(['shuf', str(united_file_path)], stdout=f, check=True)
    else:
        shuffle_with_splitting(
            united_file_path,
            shuffled_file_path,
            args.max_shuf_lines,
            args.split_dir,
            args.num_split_lines,
            num_lines,
            args.resume_from,
        )
    os.remove(united_file_path)
    for out_file in args.output_files:
        out_file.parent.mkdir(parents=True, exist_ok=True)
    output_file_objects = [out_file.open('w', buffering=BUFFER_SIZE) for out_file in args.output_files]
    with shuffled_file_path.open(buffering=BUFFER_SIZE) as f:
        for line_i, tmp_line in tqdm(enumerate(f), total=num_lines, unit='line', desc="spliting lines"):
            lines = tmp_line.strip().split(args.line_delimiter)
            assert len(lines) == len(output_file_objects), (
                f"Number of lines {len(lines)} in shuffled file {shuffled_file_path }does not equal number of output"
                f"file objects {output_file_objects}. Line from shuffled file: {repr(tmp_line)}"
            )
            for i, line in enumerate(lines):
                output_file_objects[i].write(line + '\n')
    for output_file_object in output_file_objects:
        output_file_object.close()
    os.remove(shuffled_file_path)


if __name__ == "__main__":
    main()