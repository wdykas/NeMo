import json
import os
from examples.tts.G2P.heteronyms_correction_with_classification import get_wordid_to_nemo, clean
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="Calculate accuracy of heteronyms predictions")
parser.add_argument("--manifest", type=str, help="Path to manifest files with model predictions")
parser.add_argument("--target", type=str, default=None, help="heteronym to use for accuracy calculation, the rest heteronyms will be ignored")

def _process_wiki_eval(graphemes):
    graphemes = graphemes.replace("says sorry for calling guest 'retard'", "says sorry for calling guest retard")
    return graphemes




def eval_heteronyms(manifest, target_homograph=None):
    dict_target_homograph = {}

    num_skipped = 0
    num_lines = 0
    correct = 0
    wordid_to_nemo_cmu = get_wordid_to_nemo()
    errors = []
    with open(manifest, "r") as f_in:
        for line in tqdm(f_in):
            num_lines += 1
            line = json.loads(line)
            graphemes = clean(line["text_graphemes"], do_lower=False)
            graphemes = _process_wiki_eval(graphemes)
            phonemes_preds = clean(line["pred_text"])
            phonemes_gt = clean(line["text"])

            homograph = line["homograph_span"]
            homograph_ipa = wordid_to_nemo_cmu[line["word_id"]]
            homograph_count_in_grapheme = graphemes.lower().count(homograph.lower())
            is_correct = False

            # if target_homograph is not None:
            #     if homograph != target_homograph:
            #         continue
            #     else:
            #         if homograph_ipa not in dict_target_homograph:
            #             dict_target_homograph[homograph_ipa] = 0
            #         dict_target_homograph[homograph_ipa] += 1
            #         print("=" * 40)
            #         print(graphemes)
            #         print("PRED:", phonemes_preds)
            #         print("GT  :", homograph_ipa)
            #         print("=" * 40)

            if homograph_count_in_grapheme == 1:
                if homograph_ipa in phonemes_preds:
                    is_correct = True
            elif homograph_count_in_grapheme > 1 and graphemes.count(homograph.lower()) == phonemes_preds.count(
                homograph_ipa
            ):
                is_correct = True
            else:
                if graphemes.count("-") == phonemes_gt.count("-") and graphemes.count("-") > 0:
                    graphemes = graphemes.replace("-", " - ")
                    phonemes_gt = phonemes_gt.replace("-", " - ")
                    phonemes_preds = phonemes_preds.replace("-", " - ")

                graphemes = graphemes.split()
                phonemes_preds = phonemes_preds.split()
                phonemes_gt = phonemes_gt.split()

                if graphemes.count(homograph) == 1:
                    if phonemes_preds.count(homograph_ipa) == 1:
                        is_correct = True
                elif len(graphemes) != len(phonemes_preds) or len(phonemes_preds) != len(phonemes_gt):
                    print(phonemes_preds)
                    print(phonemes_gt)
                    print(graphemes)
                    print(line["homograph_span"])

                    num_skipped += 1
                else:
                    idx = graphemes.index(line["homograph_span"])
                    if phonemes_preds[idx] == phonemes_gt[idx]:
                        is_correct = True

            if is_correct:
                correct += 1
            else:
                errors.append(line)

    errors_file = os.path.basename(manifest).replace(".json", "_errors.json")
    with open(errors_file, "w") as f:
        for entry in errors:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Errors saved in {errors_file}")

    print(
        f"{correct*100/num_lines:.2f}% -- {correct} out of {num_lines}, num_lines: {num_lines}, skipped: {num_skipped}"
    )
    print(len(dict_target_homograph), dict_target_homograph)


if __name__ == "__main__":
    args = parser.parse_args()
    # manifest = "/home/ebakhturina/NeMo/examples/tts/eval_wikihomograph_preds.json"
    #
    # manifest = "/home/ebakhturina/g2p_scripts/paper_experiments/eval_wikihomograph_g2pENG_preds.json"
    eval_heteronyms(args.manifest, args.target)
