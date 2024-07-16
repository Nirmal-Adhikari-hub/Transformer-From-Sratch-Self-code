from jamo import h2j, j2hcj
from config import (START_TOKEN, END_TOKEN, PADDING_TOKEN, MAX_LINES_IMPORT, VALID_SENTENCE_LENGTH, CSV_FILE_PATH,
                    ENGLISH_VALID_VOCAB_SENTENCES_PATH, KOREAN_VALID_VOCAB_SENTENCES_PATH)


punctuation_marks = [
    ".", ",", "?", "!", ":", ";", '"', "'", "(", ")", "-", "–", "—", "/", "\\", "…",
]

digits = [str(i) for i in range(10)]

special_symbols = [
    " ", "@", "#", "$", "%", "^", "&", "*", "_", "+", "=", "<", ">", "[", "]", "{", "}", "|", "~", "`"
]

choseong = ["ㄱ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
jungseong = ["ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ",
             "ㅢ", "ㅣ"]
jongseong = ["", "ㄲ", "ㄳ", "ㄵ", "ㄶ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅄ", "ㅆ"]

english_alphabet = [chr(i) for i in range(ord('a'), ord('z') + 1)] + [chr(i) for i in range(ord('A'), ord('Z') + 1)]

korean_vocab = [START_TOKEN, *punctuation_marks, *special_symbols, *digits, *choseong, *jungseong, *jongseong,
                PADDING_TOKEN, END_TOKEN]
english_vocab = [START_TOKEN, *english_alphabet, *special_symbols, *digits, *punctuation_marks, PADDING_TOKEN,
                 END_TOKEN]


# Path to the CSV file 


""" Extract the sentences having the vocab characters. (Comment out after running once.)"""
run_sentences_through_vocab_filtration_and_create_new_files = False
if run_sentences_through_vocab_filtration_and_create_new_files:
    # Initialize lists to store valid sentences
    korean_sentences = []
    english_sentences = []

    # Function to validate a sentence against a vocabulary
    def _is_valid_sentence(in_sentence, vocab):
        return all(char in vocab for char in list(set(in_sentence)))

    # Read the CSV file
    with open(CSV_FILE_PATH, 'r', encoding='utf-8') as file:
        lines = file.readlines()

        for i in range(0, len(lines), 2):
            korean_text = lines[i].strip().replace("Korean:", "").strip()
            english_text = lines[i + 1].strip().replace("English:", "").strip()

            # Convert Korean text using jamo functions
            korean_text_jamo = sorted(set(j2hcj(h2j(korean_text))))

            # Validate both sentences
            if _is_valid_sentence(korean_text_jamo, korean_vocab) and _is_valid_sentence(english_text, english_vocab):
                korean_sentences.append(korean_text)
                english_sentences.append(english_text)

    # Save Korean sentences to file
    with open('korean_valid_vocab_sentences.txt', 'w', encoding='utf-8') as korean_file:
        for sentence in korean_sentences:
            korean_file.write(sentence + '\n')

    # Save English sentences to file
    with open('english_valid_vocab_sentences.txt', 'w', encoding='utf-8') as english_file:
        for sentence in english_sentences:
            english_file.write(sentence + '\n')

    print("Sentences saved successfully.")


def _k_tunnel(s):
    return j2hcj(h2j(s))


def load_valid_sentences():
    with open(ENGLISH_VALID_VOCAB_SENTENCES_PATH, 'r', encoding='utf-8') as f:
        english_lines = f.readlines()[:MAX_LINES_IMPORT]

    with open(KOREAN_VALID_VOCAB_SENTENCES_PATH, 'r', encoding='utf-8') as f:
        korean_lines = f.readlines()[:MAX_LINES_IMPORT]

    english_lines = [line.rstrip('\n') for line in english_lines]
    korean_lines = [line.rstrip('\n') for line in korean_lines]

    def check_length(english, korean):
        return True if len(english) <= VALID_SENTENCE_LENGTH and len(
            _k_tunnel(korean)) <= VALID_SENTENCE_LENGTH else False

    def _filter_valid_sentences(english, korean):
        valid_english = []
        valid_korean = []
        for eng, kor in zip(english, korean):
            if check_length(eng, kor):
                valid_english.append(eng)
                valid_korean.append(kor)
        return valid_english, valid_korean

    valid_english_sentences, valid_korean_sentences = _filter_valid_sentences(english_lines, korean_lines)
    return valid_english_sentences, valid_korean_sentences


def key_value_dictionary():
    e_itos = {i: ch for i, ch in enumerate(english_vocab)}
    k_itos = {i: ch for i, ch in enumerate(korean_vocab)}
    return e_itos, k_itos


def value_key_dictionary():
    e_stoi = {ch: i for i, ch in enumerate(english_vocab)}
    k_stoi = {ch: i for i, ch in enumerate(korean_vocab)}
    return e_stoi, k_stoi

