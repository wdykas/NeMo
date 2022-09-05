import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--orig_dataset",
        "-d",
        type=Path,
        required=True,
        help="Original prompt-/p- tuning dataset in .jsonl format.",
    )
    parser.add_argument(
        "--generated_predictions",
        "-g",
        type=Path,
        required=True,
        help="'sentences' field in `generate()` method output in `megatron_gpt_prompt_learning_eval.py` script. Has to"
        "be in .json format and contains a list of model outputs.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Path to output .jsonl file.",
    )
    parser.add_argument(
        "--answer_start",
        "-a",
        help="A prefix to answer, generate by the model in output string from file `--generated_predictions`."
    )
    args = parser.parse_args()
    for k in ['orig_dataset', 'generated_predictions', 'output']:
        setattr(args, k, getattr(args, k).expanduser())
    return args


def main() -> None:
    args = parse_args()
    with args.generate_predictions.open() as gen_f:
        generated_outputs = json.load(gen_f)
    with args.orig_dataset.open() as orig_f:
        orig_lines = json.load(orig_f)
    if len(orig_lines) != len(generated_outputs):
        raise ValueError(
            f"Different number of predictions and lines in original dataset. Prediction file "
            f"{args.generated_predictions} contains {len(generated_outputs)}, whereas original dataset in file "
            f"{args.orig_dataset} contains {len(orig_lines)}."
        )
    answer_key = "model_answer"
    with args.output.open() as out_f:
        for i, (orig_line, generated_output) in enumerate(zip(orig_lines, generated_outputs)):
            orig_datum = json.loads(orig_line)
            if args.answer_start not in generated_output:
                raise ValueError(
                    f"Answer prompt '{args.answer_start}' is not found in generated output number {i}. "
                    f"args.answer_start='{args.answer_start}', generated_output='{generated_output}'."
                )
            count = generated_output.count(args.answer_start)
            if count > 1:
                raise ValueError(
                    f"Found {count} answer prompts '{args.answer_start}' in generated output number {i}. "
                    f"args.answer_start='{args.answer_start}', generated_output='{generated_output}'."
                )
            if answer_key in orig_datum:
                raise ValueError(
                    f"Answer key '{answer_key}' is already present in a dataset sample number {i} '{orig_datum}'."
                )
            answer = generated_output.split(args.answer_start)[-1]
            orig_datum[answer_key] = answer
            out_f.write(json.dumps(orig_datum, indent=2) + '\n')


if __name__ == "__main__":
    main()
