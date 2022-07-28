import json
import os
from glob import glob

from nemo_text_processing.g2p.data.data_utils import correct_wikihomograph_data, read_wikihomograph_file
from tqdm import tqdm


def convert_wikihomograph_data_to_manifest(data_folder: str, output_manifest: str):
    """
    Reads WikiHomograph data, where the ,
    see https://github.com/google-research-datasets/WikipediaHomographData/tree/master/data/eval for the format details

    Args:
        data_folder: data_folder that contains .tsv files
        output_manifest: path to output file
    """
    with open(output_manifest, "w") as f_out:
        for file in tqdm(glob(f"{data_folder}/*.tsv")):
            file_name = os.path.basename(file)
            sentences, start_end_indices, homographs, word_ids = read_wikihomograph_file(file)
            for i, sent in enumerate(sentences):
                start, end = start_end_indices[i]
                sent, start, end = correct_wikihomograph_data(sent, start, end)
                homograph = file_name.replace(".tsv", "")

                homograph_span = sent[start:end]
                if homograph_span.lower() != homograph and sent.lower().count(homograph) == 1:
                    start = sent.lower().index(homograph)
                    end = start + len(homograph)
                    homograph_span = sent[start:end].lower()
                    assert homograph == homograph_span.lower()

                assert homograph_span.lower() == homograph
                entry = {
                    "text_graphemes": sent,
                    "start_end": [start, end],
                    "homograph_span": homograph_span,
                    "word_id": word_ids[i],
                }
                f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Data saved at {output_manifest}")


if __name__ == '__main__':
    convert_wikihomograph_data_to_manifest(
        "/home/ebakhturina/g2p_scripts/WikipediaHomographData-master/data/train", "train.json"
    )
    convert_wikihomograph_data_to_manifest(
        "/home/ebakhturina/g2p_scripts/WikipediaHomographData-master/data/eval", "eval.json"
    )
