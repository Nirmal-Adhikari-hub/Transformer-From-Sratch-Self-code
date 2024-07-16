import torch

TRAIN =  True

BATCH_SIZE = 5
NUM_EPOCHS = 5

DROPOUT = 0.1
START_TOKEN = '<START>'
END_TOKEN = '<END>'
PADDING_TOKEN = '<PADDING>'
PERCENTILE = 98
BLOCK_SIZE = 200
LR = 1 # Learning Rate
LOG_INTERVAL = 2 # Batch interval to print training stastistics
NUM_TOKEN_GEN = 30

''' Architecture Paramaters'''
NUM_HEADS = 8
N_EMBD = 512  # (d_model)
NUM_LAYERS = 6

# Global variables
VALID_SENTENCE_LENGTH = 295
MAX_LINES_IMPORT = 500
CSV_FILE_PATH = '../../korean_english_ted_talks.csv'
ENGLISH_VALID_VOCAB_SENTENCES_PATH = 'english_valid_vocab_sentences.txt'
KOREAN_VALID_VOCAB_SENTENCES_PATH = 'korean_valid_vocab_sentences.txt'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Automatically return all variables defined in this module
__all__ = list(globals().keys())

 