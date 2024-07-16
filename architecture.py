import torch
import torch.nn as nn
from torch.nn import functional as F
from config import BLOCK_SIZE, BATCH_SIZE, DROPOUT, NUM_HEADS, N_EMBD, NUM_LAYERS
from preprocessing import key_value_dictionary, value_key_dictionary


e_itos, k_itos = key_value_dictionary()
e_stoi, k_stoi = value_key_dictionary()
VOCAB_SIZE_K = len(k_itos) if len(k_itos) == len(k_stoi) else print("Key-Value and Vaue-Key dictionary don't match.")
VOCAB_SIZE_E = len(e_itos) if len(e_itos) == len(k_stoi) else print("Key-Value and Vaue-Key dictionary don't match.")


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, N_EMBD)


    def forward(self, x):
        # X (B, T)
        token_embd = self.token_embedding(x) # (B, T, C)
        pos_emb = self.position_embedding(torch.arange(x.shape[-1], device=x.device)) # (T, C)
        # print(f"EMBEDDING:{(token_embd + pos_emb).shape}")
        return token_embd + pos_emb



class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.ReLU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        # X shape (B, T, C)
        return self.net(x)



class Head(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.key = nn.Linear(N_EMBD, head_dim, bias=False)
        self.query = nn.Linear(N_EMBD, head_dim, bias=False)
        self.value = nn.Linear(N_EMBD, head_dim, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x, pad_mask=None, encoder_out=None, mask=False):
        
        B, T, C = x.shape
        k = self.key(encoder_out if encoder_out is not None else x)
        v = self.value(encoder_out if encoder_out is not None else x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        # print(f"ENCODER_OUT -> {encoder_out.shape if encoder_out is not None else f'NO ENCODER_OUT'},Q {q.shape}, K {k.shape}, V: {v.shape}")
        # print(f"Self Attention: x -> {x.shape}, pad_mask -> {pad_mask}, wei -> {wei.shape}")
        if mask:
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        if pad_mask is not None:
            if not isinstance(pad_mask, torch.Tensor):
                pad_mask = torch.tensor(pad_mask, dtype=torch.bool, device=x.device)
            if pad_mask.dim() == 2:  # Ensure pad_mask has 3 dimensions
                pad_mask = pad_mask.unsqueeze(1)
                # print(f"Pad_mask after unsqueeze shape -> {pad_mask.shape}")
            wei = wei.masked_fill(pad_mask == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out



class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = N_EMBD // NUM_HEADS
        self.heads = nn.ModuleList([Head(self.head_dim) for _ in range(NUM_HEADS)])
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)


    def forward(self, x, pad_mask=None, encoder_out=None, mask=False):
        # X shape (B, T, C)
        # print(f"Multi HA: x -> {x.shape}, pad_mask -> {pad_mask}")
        out = torch.cat([h(x, pad_mask, encoder_out, mask) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out # (B, T, C)



class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.att = MultiHeadAttention()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.ffn = FeedForward()
        self.ln2 = nn.LayerNorm(N_EMBD)


    def forward(self, x, pad_mask):
        # print(f"ENCODER LAYER: X -> {x.shape}, pad-mask -> {pad_mask.shape}")
        x = x + self.att(self.ln1(x), pad_mask=pad_mask)
        x = x + self.ffn(self.ln2(x))
        return x



class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = EmbeddingLayer(VOCAB_SIZE_E)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(NUM_LAYERS)])


    def forward(self, x, pad_mask):
        # print(f"ENCODER x -> {x.shape}, pad_mask -> {pad_mask.shape}")
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, pad_mask=pad_mask)
        return x



class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = MultiHeadAttention()
        self.cross_a = MultiHeadAttention()
        self.ffn = FeedForward()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.ln3 = nn.LayerNorm(N_EMBD)


    def forward(self, y, encoder_out, pad_mask, mask=True):
        y = y + self.sa(self.ln1(y), encoder_out=None, pad_mask=pad_mask, mask=True)
        # print("SELF AT COMPLETED \n\n\n")
        y = y + self.cross_a(self.ln2(y), encoder_out=encoder_out, pad_mask=pad_mask, mask=False)
        # print("CROSS AT COMPLETED \n")
        y = y + self.ffn(self.ln3(y))
        return y
    


class DecoderBlock(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.embedding = EmbeddingLayer(VOCAB_SIZE_K)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(NUM_LAYERS)])
        self.out = nn.Linear(N_EMBD, VOCAB_SIZE_K)
        

    def forward(self, y, encoder_out, pad_mask):
        y = self.embedding(y)
        # print(f"DECODER BLOCK after embedding {y.shape}")
        for layer in self.layers:
            y = layer(y, encoder_out=encoder_out, pad_mask=pad_mask)
        return self.out(y) 