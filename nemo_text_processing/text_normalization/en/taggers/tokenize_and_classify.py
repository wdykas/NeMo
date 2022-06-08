# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_WHITE_SPACE,
    GraphFst,
    delete_extra_space,
    delete_space,
    generator_main,
    NEMO_NOT_SPACE,
    NEMO_SIGMA
)
from nemo_text_processing.text_normalization.en.taggers.abbreviation import AbbreviationFst
from nemo_text_processing.text_normalization.en.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.en.taggers.date import DateFst
from nemo_text_processing.text_normalization.en.taggers.decimal import DecimalFst
from nemo_text_processing.text_normalization.en.taggers.electronic import ElectronicFst
from nemo_text_processing.text_normalization.en.taggers.fraction import FractionFst
from nemo_text_processing.text_normalization.en.taggers.measure import MeasureFst
from nemo_text_processing.text_normalization.en.taggers.money import MoneyFst
from nemo_text_processing.text_normalization.en.taggers.ordinal import OrdinalFst
from nemo_text_processing.text_normalization.en.taggers.punctuation import PunctuationFst
from nemo_text_processing.text_normalization.en.taggers.range import RangeFst as RangeFst
from nemo_text_processing.text_normalization.en.taggers.roman import RomanFst
from nemo_text_processing.text_normalization.en.taggers.serial import SerialFst
from nemo_text_processing.text_normalization.en.taggers.telephone import TelephoneFst
from nemo_text_processing.text_normalization.en.taggers.time import TimeFst
from nemo_text_processing.text_normalization.en.taggers.whitelist import WhiteListFst
from nemo_text_processing.text_normalization.en.taggers.word import WordFst
from nemo_text_processing.text_normalization.en.verbalizers.cardinal import CardinalFst as vCardinal
from nemo_text_processing.text_normalization.en.verbalizers.date import DateFst as vDate
from nemo_text_processing.text_normalization.en.verbalizers.decimal import DecimalFst as vDecimal
from nemo_text_processing.text_normalization.en.verbalizers.electronic import ElectronicFst as vElectronic
from nemo_text_processing.text_normalization.en.verbalizers.fraction import FractionFst as vFraction
from nemo_text_processing.text_normalization.en.verbalizers.measure import MeasureFst as vMeasure
from nemo_text_processing.text_normalization.en.verbalizers.money import MoneyFst as vMoney
from nemo_text_processing.text_normalization.en.verbalizers.ordinal import OrdinalFst as vOrdinal
from nemo_text_processing.text_normalization.en.verbalizers.roman import RomanFst as vRoman
from nemo_text_processing.text_normalization.en.verbalizers.telephone import TelephoneFst as vTelephone
from nemo_text_processing.text_normalization.en.verbalizers.time import TimeFst as vTime
from nemo_text_processing.text_normalization.en.verbalizers.word import WordFst as vWord

from nemo.utils import logging

try:
    import pynini
    from pynini.lib import pynutil
    from pynini.examples import plurals

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False

logging.setLevel("INFO")


class ClassifyFst(GraphFst):
    """
    Final class that composes all other classification grammars. This class can process an entire sentence including punctuation.
    For deployment, this grammar will be compiled and exported to OpenFst Finate State Archiv (FAR) File. 
    More details to deployment at NeMo/tools/text_processing_deployment.
    
    Args:
        input_case: accepting either "lower_cased" or "cased" input.
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
        cache_dir: path to a dir with .far grammar file. Set to None to avoid using cache.
        overwrite_cache: set to True to overwrite .far files
        whitelist: path to a file with whitelist replacements
    """

    def __init__(
        self,
        input_case: str,
        deterministic: bool = True,
        cache_dir: str = None,
        overwrite_cache: bool = False,
        whitelist: str = None,
    ):
        super().__init__(name="tokenize_and_classify", kind="classify", deterministic=deterministic)

        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            whitelist_file = os.path.basename(whitelist) if whitelist else ""
            far_file = os.path.join(
                cache_dir, f"en_tn_{deterministic}_deterministic_{input_case}_{whitelist_file}_tokenize_single_graph.far"
            )
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
            logging.info(f'ClassifyFst.fst was restored from {far_file}.')
        else:
            logging.info(f"Creating ClassifyFst grammars.")
            start_time = time.time()
            cardinal = CardinalFst(deterministic=deterministic)
            cardinal_graph = cardinal.fst
            logging.info(f"cardinal: {time.time() - start_time: .2f}s -- {cardinal_graph.num_states()} nodes")

            start_time = time.time()
            ordinal = OrdinalFst(cardinal=cardinal, deterministic=deterministic)
            ordinal_graph = ordinal.fst
            logging.info(f"ordinal: {time.time() - start_time: .2f}s -- {ordinal_graph.num_states()} nodes")

            start_time = time.time()
            decimal = DecimalFst(cardinal=cardinal, deterministic=deterministic)
            decimal_graph = decimal.fst
            logging.info(f"decimal: {time.time() - start_time: .2f}s -- {decimal_graph.num_states()} nodes")

            start_time = time.time()
            fraction = FractionFst(deterministic=deterministic, cardinal=cardinal)
            fraction_graph = fraction.fst
            logging.info(f"fraction: {time.time() - start_time: .2f}s -- {fraction_graph.num_states()} nodes")

            start_time = time.time()
            measure = MeasureFst(cardinal=cardinal, decimal=decimal, fraction=fraction, deterministic=deterministic)
            measure_graph = measure.fst
            logging.info(f"measure: {time.time() - start_time: .2f}s -- {measure_graph.num_states()} nodes")

            start_time = time.time()
            date = DateFst(cardinal=cardinal, deterministic=deterministic, lm=True)
            date_graph = date.fst
            logging.info(f"date: {time.time() - start_time: .2f}s -- {date_graph.num_states()} nodes")

            start_time = time.time()
            time_graph = TimeFst(cardinal=cardinal, deterministic=deterministic).fst
            logging.info(f"time: {time.time() - start_time: .2f}s -- {time_graph.num_states()} nodes")

            start_time = time.time()
            telephone_graph = TelephoneFst(deterministic=deterministic).fst
            logging.info(f"telephone: {time.time() - start_time: .2f}s -- {telephone_graph.num_states()} nodes")

            start_time = time.time()
            electronic_graph = ElectronicFst(deterministic=deterministic).fst
            logging.info(f"electronic: {time.time() - start_time: .2f}s -- {electronic_graph.num_states()} nodes")

            start_time = time.time()
            money_graph = MoneyFst(cardinal=cardinal, decimal=decimal, deterministic=False).fst
            logging.info(f"money: {time.time() - start_time: .2f}s -- {money_graph.num_states()} nodes")

            start_time = time.time()
            whitelist_graph = WhiteListFst(
                input_case=input_case, deterministic=deterministic, input_file=whitelist
            ).fst
            logging.info(f"whitelist: {time.time() - start_time: .2f}s -- {whitelist_graph.num_states()} nodes")

            start_time = time.time()
            punctuation = PunctuationFst(deterministic=deterministic)
            punct_graph = punctuation.graph
            logging.info(f"punct: {time.time() - start_time: .2f}s -- {punct_graph.num_states()} nodes")

            start_time = time.time()
            word_graph = WordFst(punctuation=punctuation, deterministic=deterministic).graph
            logging.info(f"word: {time.time() - start_time: .2f}s -- {word_graph.num_states()} nodes")

            start_time = time.time()
            serial_graph = SerialFst(cardinal=cardinal, ordinal=ordinal, deterministic=deterministic).fst
            logging.info(f"serial: {time.time() - start_time: .2f}s -- {serial_graph.num_states()} nodes")

            start_time = time.time()

            cardinal_verbalizer = vCardinal(deterministic=deterministic)
            v_cardinal_graph = cardinal_verbalizer.fst
            decimal = vDecimal(cardinal=cardinal_verbalizer, deterministic=deterministic)
            v_decimal_graph = decimal.fst
            ordinal = vOrdinal(deterministic=deterministic)
            v_ordinal_graph = ordinal.fst
            fraction = vFraction(deterministic=deterministic)
            v_fraction_graph = fraction.fst
            v_telephone_graph = vTelephone(deterministic=deterministic).fst
            v_electronic_graph = vElectronic(deterministic=deterministic).fst
            measure = vMeasure(decimal=decimal, cardinal=cardinal_verbalizer, fraction=fraction, deterministic=deterministic)
            v_measure_graph = measure.fst
            v_time_graph = vTime(deterministic=deterministic).fst
            v_date_graph = vDate(ordinal=ordinal, deterministic=deterministic).fst
            v_money_graph = vMoney(decimal=decimal, deterministic=False).fst
            v_roman_graph = vRoman(deterministic=deterministic).fst
            v_word_graph = vWord(deterministic=deterministic).fst


            
            logging.info(f"verbalization done")

            cardinal_or_date_final = plurals._priority_union(date_graph, cardinal_graph, NEMO_SIGMA)
            cardinal_or_date_final = pynini.compose(cardinal_or_date_final, (v_cardinal_graph | v_date_graph))

            time_final = pynini.compose(time_graph, v_time_graph)
            ordinal_final = pynini.compose(ordinal_graph, v_ordinal_graph)

            sem_w = 1
            word_w = 100
            punct_w = 2
            
            classify_and_verbalize = (
                pynutil.add_weight(time_final, sem_w)
                | pynutil.add_weight(pynini.compose(decimal_graph, v_decimal_graph), sem_w)
                | pynutil.add_weight(pynini.compose(measure_graph, v_measure_graph), sem_w)
                | pynutil.add_weight(ordinal_final, sem_w)
                | pynutil.add_weight(pynini.compose(telephone_graph, v_telephone_graph), sem_w)
                | pynutil.add_weight(pynini.compose(electronic_graph, v_electronic_graph), sem_w)
                | pynutil.add_weight(pynini.compose(fraction_graph, v_fraction_graph), sem_w)
                | pynutil.add_weight(pynini.compose(money_graph, v_money_graph), sem_w)
                | pynutil.add_weight(cardinal_or_date_final, sem_w)
                | pynutil.add_weight(whitelist_graph, sem_w)
                | pynutil.add_weight(
                    pynini.compose(serial_graph, v_word_graph), 1.1001
                )  # should be higher than the rest of the classes
            ).optimize()


            logging.info(f"final done")
            roman_graph = RomanFst(deterministic=deterministic).fst
            classify_and_verbalize |= pynutil.add_weight(pynini.compose(roman_graph, v_roman_graph), sem_w)

            date_final = pynini.compose(date_graph, v_date_graph)
            range_graph = RangeFst(
                time=time_final, cardinal=cardinal, date=date_final, deterministic=deterministic
            ).fst
            classify_and_verbalize |= pynutil.add_weight(pynini.compose(range_graph, v_word_graph), sem_w)
            # classify_and_verbalize = pynutil.insert("< ") + classify_and_verbalize + pynutil.insert(" >")
            classify_and_verbalize |= pynutil.add_weight(word_graph, word_w)

            punct_only = pynutil.add_weight(punct_graph, weight=punct_w)
            punct = pynini.closure(
                pynini.compose(pynini.closure(NEMO_WHITE_SPACE, 1), delete_extra_space)
                | (pynutil.insert(" ") + punct_only),
                1,
            )

            def get_token_sem_graph(classify_and_verbalize):
                token_plus_punct = (
                    pynini.closure(punct + pynutil.insert(" "))
                    + classify_and_verbalize
                    + pynini.closure(pynutil.insert(" ") + punct)
                )

                graph = token_plus_punct + pynini.closure(
                    (
                        pynini.compose(pynini.closure(NEMO_WHITE_SPACE, 1), delete_extra_space)
                        | (pynutil.insert(" ") + punct + pynutil.insert(" "))
                    )
                    + token_plus_punct
                )

                graph |= punct_only + pynini.closure(punct)
                graph = delete_space + graph + delete_space

                remove_extra_spaces = pynini.closure(NEMO_NOT_SPACE, 1) + pynini.closure(
                    delete_extra_space + pynini.closure(NEMO_NOT_SPACE, 1)
                )
                remove_extra_spaces |= (
                    pynini.closure(pynutil.delete(" "), 1)
                    + pynini.closure(NEMO_NOT_SPACE, 1)
                    + pynini.closure(delete_extra_space + pynini.closure(NEMO_NOT_SPACE, 1))
                )

                graph = pynini.compose(graph.optimize(), remove_extra_spaces).optimize()
                return graph

            self.fst = get_token_sem_graph(classify_and_verbalize)

            self.fst = self.fst.optimize()

            if far_file:
                generator_main(far_file, {"tokenize_and_classify": self.fst})
                logging.info(f"ClassifyFst grammars are saved to {far_file}.")
            import ipdb; ipdb.set_trace()
