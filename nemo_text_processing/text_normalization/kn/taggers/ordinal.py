# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.kn.graph_utils import GraphFst
from nemo_text_processing.text_normalization.kn.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.kn.utils import get_abs_path


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying Kannada ordinals, e.g.
        1ನೇ -> ordinal { integer: "ಮೊದಲನೆಯ" }
        ೧೦ನೇ -> ordinal { integer: "ಹತ್ತನೆಯ" }
        22ನೇ -> ordinal { integer: "ಇಪ್ಪತ್ತೆರಡನೆಯ" }
        2nd -> ordinal { integer: "ಎರಡನೆಯ" }
        
    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: CardinalFst, deterministic: bool = True):
        super().__init__(name="ordinal", kind="classify", deterministic=deterministic)

        # Load suffix list and exceptions
        suffixes = pynini.string_file(get_abs_path("data/ordinal/suffixes.tsv"))
        exceptions = pynini.string_file(get_abs_path("data/ordinal/exceptions.tsv"))
        
        # Except 1 take all digits , as 1 is in exceptions and is handled separately
        digit_acceptor=pynini.project(cardinal.digit,"input")
        digit_no_one=pynini.difference(digit_acceptor,pynini.union("೧","1")).optimize()
        digit_no_one=pynini.compose(digit_no_one, cardinal.digit).optimize()

        # Build limited cardinal graph (up to ten thousands for practical ordinals)
        limited_cardinal_graph = (
            digit_no_one
            | cardinal.zero
            | cardinal.teens_and_ties
            | cardinal.graph_hundreds
            | cardinal.graph_thousands
            | cardinal.graph_ten_thousands
        ).optimize()

        # Ordinal suffix transformation:
        # Cardinal words end in ು (e.g., ಎರಡು, ಹತ್ತು, ನೂರು)
        # Ordinal form: strip ು, add ನೆಯ (e.g., ಎರಡನೆಯ, ಹತ್ತನೆಯ, ನೂರನೆಯ)
        # Sigma includes Kannada characters and space (for compound cardinals like ನೂರ ಒಂದು)
        kannada_chars = pynini.union(*[chr(c) for c in range(0x0C80, 0x0CFF)])
        sigma = pynini.closure(kannada_chars | pynini.accep(" "))

        kn_suffix = pynutil.delete(suffixes)
        en_suffix = pynutil.delete(pynini.union("st", "nd", "rd", "th"))
        suffix= kn_suffix | en_suffix
        
        # Combined rewrite rule for all endings
        # ు -> ನೆಯ (most cardinals: ಒಂದು, ಎರಡು, ಹತ್ತು, ನೂರು, etc.)
        # ರ -> ರನೆಯ (ಸಾವಿರ)
        # ಷ -> ಷನೆಯ (ಲಕ್ಷ)  
        # ಿ -> ಿಯನೆಯ (ಕೋಟಿ)
        ordinal_rewrite = pynini.union(
            pynini.cross("ು", "ನೆಯ"),
            pynini.cross("ರ", "ರನೆಯ"),
            pynini.cross("ಷ", "ಷನೆಯ"),
            pynini.cross("ಿ", "ಿಯನೆಯ"),
        )
        #sigma          → matches "ಇಪ್ಪತ್ತೈದ"  (everything before last)
        #ordinal_rewrite → matches "ು" → "ನೆಯ"  (only the end)
        #result → "ಇಪ್ಪತ್ತೈದನೆಯ"
        cardinal_to_ordinal = sigma + ordinal_rewrite

        # cardinal → Kannada word → apply end transform → delete suffix
        ordinal_graph = (limited_cardinal_graph @ cardinal_to_ordinal) + suffix

        # Exceptions must ALSO consume the suffix
        exceptions_graph = pynutil.add_weight(exceptions, -0.1)
        
        #Final union
        graph = pynini.union(
           exceptions_graph,
           ordinal_graph,
        ).optimize()

        self.graph = graph.optimize()

        final_graph = pynutil.insert("integer: \"") + graph + pynutil.insert("\"")
        final_graph = self.add_tokens(final_graph)

        self.fst = final_graph.optimize()