from preprocessing import load_data, key_value_dictionary, value_key_dictionary
from jamo import h2j, j2hcj
from config import PERCENTILE
import numpy as np


DECODE = True

english_sentences, korean_sentences = load_data()
e_itos, k_itos = key_value_dictionary()
e_stoi, k_stoi = value_key_dictionary()


def _k_tunnel(sentence):
    return j2hcj(h2j(sentence))


# print(max(len(sentence) for sentence in english_lines), max(len(k_tunnel(sentence)) for sentence in korean_lines))


if DECODE:
    # print(f"{PERCENTILE}th percentile length English:{np.percentile([len(x) for x in english_lines], PERCENTILE)}")
    # print(f"{PERCENTILE}th percentile length Korean:"
    #       f"{np.percentile([len(_k_tunnel(x)) for x in korean_lines], PERCENTILE)}")
    print(len(english_sentences), len(korean_sentences))
    print(english_sentences[5:13], korean_sentences[5:13])
