import json
import sys

from g2p_ctc_inference import clean, get_wordid_to_nemo
from tqdm import tqdm


def _process_wiki_eval(graphemes):
    graphemes = graphemes.replace("says sorry for calling guest 'retard'", "says sorry for calling guest retard")
    return graphemes


def eval_heteronyms(manifest):
    num_skipped = 0
    num_lines = 0
    correct = 0
    wordid_to_nemo_cmu = get_wordid_to_nemo()
    with open(manifest, "r") as f_in:
        for line in tqdm(f_in):
            num_lines += 1
            line = json.loads(line)
            graphemes = clean(line["text_graphemes"], do_lower=False)
            graphemes = _process_wiki_eval(graphemes)
            phonemes_preds = line["pred_text"]
            phonemes_gt = line["text"]

            homograph = line["homograph_span"]
            homograph_ipa = wordid_to_nemo_cmu[line["word_id"]]
            homograph_count_in_grapheme = graphemes.lower().count(homograph.lower())
            if homograph_count_in_grapheme == 1:
                if homograph_ipa in phonemes_preds:
                    correct += 1
            elif homograph_count_in_grapheme > 1 and graphemes.count(homograph.lower()) == phonemes_preds.count(
                homograph_ipa
            ):
                correct += 1
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
                        correct += 1
                elif len(graphemes) != len(phonemes_preds) or len(phonemes_preds) != len(phonemes_gt):
                    print(phonemes_preds)
                    print(phonemes_gt)
                    print(graphemes)
                    print(line["homograph_span"])

                    num_skipped += 1
                    import pdb

                    pdb.set_trace()
                    print()
                else:
                    try:
                        idx = graphemes.index(line["homograph_span"])
                    except:
                        print(line)
                        import pdb

                        pdb.set_trace()
                        print()
                    if phonemes_preds[idx] == phonemes_gt[idx]:
                        correct += 1

    print(
        f"{correct*100/num_lines:.2f}% -- {correct} out of {num_lines}, num_lines: {num_lines}, skipped: {num_skipped}"
    )


if __name__ == "__main__":

    manifest = sys.argv[1]
    # manifest = "/home/ebakhturina/NeMo/examples/tts/eval_wikihomograph_preds.json"
    #
    # manifest = "/home/ebakhturina/g2p_scripts/paper_experiments/eval_wikihomograph_g2pENG_preds.json"
    eval_heteronyms(manifest)
