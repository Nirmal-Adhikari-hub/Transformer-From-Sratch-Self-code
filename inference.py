import torch
from hangul_jamo import compose
from model import Transformer
from config import START_TOKEN, END_TOKEN, PADDING_TOKEN, DEVICE, BLOCK_SIZE, NUM_TOKEN_GEN
from architecture import e_itos, e_stoi, k_itos, k_stoi


def preprocess_sentence(sentence, src_vocab, block_size):
    """
    Convert a sentence into a tensor of token indices and pad it to the block size.
    """
    tokens = [src_vocab.get(ch, src_vocab[PADDING_TOKEN]) for ch in sentence]
    tokens = tokens[:block_size]  # Truncate
    tokens += [src_vocab[PADDING_TOKEN]] * (block_size - len(tokens))  # Pad
    return torch.tensor(tokens).unsqueeze(0).to(DEVICE)  # (1, block_size)


def postprocess_output(output_tensor, tgt_vocab):
    """
    Convert the tensor of token indices back to a string.
    """
    tokens = output_tensor.squeeze(0).tolist()
    sentence = compose(''.join([tgt_vocab[idx] for idx in tokens if idx in tgt_vocab]))
    return sentence


def generate_translation(model, src_tensor, src_pad_mask, k_stoi, k_itos):
    # Initialize the target sequence with the start token
    tgt_tensor = torch.tensor([[k_stoi[START_TOKEN]]], dtype=torch.long).to(DEVICE)
    tgt_pad_mask = torch.ones_like(tgt_tensor).to(DEVICE)
    
    for _ in range(NUM_TOKEN_GEN):  # Arbitrary max length
        output = model(src_tensor, tgt_tensor, src_pad_mask=src_pad_mask, tgt_pad_mask=None)
        next_token = output.argmax(dim=-1)[:,-1]  # Get the index of the highest probability token
        tgt_tensor = torch.cat((tgt_tensor, next_token.unsqueeze(1)), dim=1)  # Append next_token to tgt_tensor
        # tgt_pad_mask = torch.cat((tgt_pad_mask, torch.ones_like(next_token.unsqueeze(1))), dim=1)  # Update tgt_pad_mask
        
        if next_token.item() == k_stoi[END_TOKEN]:  # Check if end token is generated
            break
    
    return tgt_tensor


def main(input_sentence):
    model = Transformer().to(DEVICE)
    model.load_state_dict(torch.load('English_to_korean.pth', map_location=DEVICE))
    
    src_tensor = preprocess_sentence(input_sentence, e_stoi, BLOCK_SIZE)
    src_pad_mask = (src_tensor != e_stoi[PADDING_TOKEN]).float()
    
    tgt_tensor = generate_translation(model, src_tensor, src_pad_mask, k_stoi, k_itos)
    translation = postprocess_output(tgt_tensor, k_itos)
    
    print(f"Input: {input_sentence}")
    print(f"Translation: {translation}")


input_sentence = "And we're going to 5000 you some stories from the sea here in video."  # Replace this with any input sentence you want to translate
main(input_sentence)
