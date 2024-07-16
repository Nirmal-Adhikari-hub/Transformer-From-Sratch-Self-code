from preprocessing import load_valid_sentences, key_value_dictionary, value_key_dictionary
import hangul_jamo
from config import PERCENTILE, BLOCK_SIZE, VALID_SENTENCE_LENGTH, PADDING_TOKEN, START_TOKEN, END_TOKEN
import torch


DECODE = False

english_sentences, korean_sentences = load_valid_sentences()
e_itos, k_itos = key_value_dictionary()
e_stoi, k_stoi = value_key_dictionary()


def k_tunnel(sentence):
    return hangul_jamo.decompose(sentence)


# print(max(len(sentence) for sentence in english_lines), max(len(k_tunnel(sentence)) for sentence in korean_lines))

def _pad_sequences(sequences, is_korean=False):
    padded_sequences = []
    for seq in sequences:
        if is_korean:
            seq = list(hangul_jamo.decompose(seq))
            # Add start and end tokens for Korean sequences
            seq = [START_TOKEN] + seq + [END_TOKEN]
        seq = list(seq)  # Convert sentence to list of characters
        if len(seq) < VALID_SENTENCE_LENGTH:
            padded_seq = seq + [PADDING_TOKEN] * (VALID_SENTENCE_LENGTH - len(seq))
        else:
            padded_seq = seq[:VALID_SENTENCE_LENGTH]
        padded_sequences.append(padded_seq)
    return padded_sequences


def _tokenize(sequences, v_to_k): 
    tokenized_sequences = []
    for sequence in sequences:
        tokenized_sequence = [v_to_k[ch] for ch in sequence if ch in v_to_k]
        # Ensure sequence length is not longer than max_length
        tokenized_sequence = tokenized_sequence[:VALID_SENTENCE_LENGTH]
        tokenized_sequences.append(tokenized_sequence)
    return tokenized_sequences


def return_tokenized_sentences():
    # Pad the sentences
    padded_english_sentences = _pad_sequences(english_sentences)
    padded_korean_sentences = _pad_sequences(korean_sentences, is_korean=True)

    # Tokenize the padded sentences
    tokenized_english_sentences = _tokenize(padded_english_sentences, e_stoi)
    tokenized_korean_sentences = _tokenize(padded_korean_sentences, k_stoi)

    # Convert to torch tensors
    tokenized_english_sentences = torch.tensor(tokenized_english_sentences)
    tokenized_korean_sentences = torch.tensor(tokenized_korean_sentences)
    return tokenized_english_sentences, tokenized_korean_sentences


if DECODE:
    pass
    # print(f"{PERCENTILE}th percentile length English:{np.percentile([len(x) for x in english_sentences],
    #                                                                 PERCENTILE)}")
    # print(f"{PERCENTILE}th percentile length Korean:"
    #       f"{np.percentile([len(_k_tunnel(x)) for x in korean_sentences], PERCENTILE)}")
    # print(len(tokenized_english_sentences), len(tokenized_korean_sentences))
    # print(english_sentences[5:13], korean_sentences[5:13])
    # print(tokenized_english_sentences[5:13], tokenized_korean_sentences[5:13])



