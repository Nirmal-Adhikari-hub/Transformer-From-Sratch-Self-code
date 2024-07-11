from preprocessing import load_valid_sentences, key_value_dictionary, value_key_dictionary
from jamo import h2j, j2hcj
from config import PERCENTILE
import numpy as np


DECODE = True

english_sentences, korean_sentences = load_valid_sentences()
e_itos, k_itos = key_value_dictionary()
e_stoi, k_stoi = value_key_dictionary()


def _k_tunnel(sentence):
    return j2hcj(h2j(sentence))


# print(max(len(sentence) for sentence in english_lines), max(len(k_tunnel(sentence)) for sentence in korean_lines))

def tokenize(sequences, v_to_k, is_korean=False):
    tokenized_sequences = []
    for sequence in sequences:
        if is_korean:
            sequence = _k_tunnel(sequence)
        tokenized_sequence = [v_to_k[ch] for ch in sequence if ch in v_to_k]
        tokenized_sequences.append(tokenized_sequence)
    return tokenized_sequences


if DECODE:
    # print(f"{PERCENTILE}th percentile length English:{np.percentile([len(x) for x in english_lines], PERCENTILE)}")
    # print(f"{PERCENTILE}th percentile length Korean:"
    #       f"{np.percentile([len(_k_tunnel(x)) for x in korean_lines], PERCENTILE)}")
    print(len(english_sentences), len(korean_sentences))
    print(english_sentences[5:13], korean_sentences[5:13])
    tokenized_english_sentences = tokenize(english_sentences, e_stoi)
    print(tokenized_english_sentences[5:13])




