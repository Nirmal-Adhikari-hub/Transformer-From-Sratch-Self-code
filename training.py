import hangul_jamo
from tqdm import tqdm
import time
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from preprocessing import key_value_dictionary, value_key_dictionary
from tokenization import return_tokenized_sentences, k_tunnel
import torch
from torch.utils.data import Dataset, DataLoader
from config import BLOCK_SIZE, PADDING_TOKEN, DEVICE, BATCH_SIZE, NUM_EPOCHS, LR, LOG_INTERVAL, TRAIN
from model import Transformer
from sklearn.model_selection import train_test_split

e_itos, k_itos = key_value_dictionary()
e_stoi, k_stoi = value_key_dictionary()
VOCAB_SIZE_K = len(k_itos) if len(k_itos) == len(k_stoi) else print("Key-Value and Vaue-Key dictionary don't match.")
VOCAB_SIZE_E = len(e_itos) if len(e_itos) == len(k_stoi) else print("Key-Value and Vaue-Key dictionary don't match.")
english_tokens, korean_tokens = return_tokenized_sentences()


class TextDataset(Dataset):
    def __init__(self, source_tokens, target_tokens, block_size):
        self.source_tokens = source_tokens # (number_of_sentences, VALID_SENTENCE_L
        self.target_tokens = target_tokens # (number_of_sentences, VALID_SENTENCE_LENGTH)
        self.block_size = block_size


    def __len__(self):
        return len(self.source_tokens)
    

    def __getitem__(self, idx):
        source_sequence = self.source_tokens[idx] # (VALID_SENTENCE_LENGTH, )
        target_sequence = self.target_tokens[idx] # (VALID_SENTENCE_LENGTH, )

        # Truncate and pad the source sequence (BLOCK_SIZE, )
        source_sequence = torch.cat((source_sequence, torch.tensor([e_stoi[PADDING_TOKEN]] * (self.block_size - 
        len(source_sequence))))) if len(source_sequence) < self.block_size else source_sequence[:self.block_size]

        # Truncate and pad the target sequence (BLOCK_SIZE, )
        target_sequence = torch.cat((target_sequence, torch.tensor([k_stoi[PADDING_TOKEN]] * (self.block_size - 
        len(target_sequence))))) if len(target_sequence) < self.block_size else target_sequence[:self.block_size]

        source_mask = (source_sequence != e_stoi[PADDING_TOKEN]).float() # (BLOCK_SIZE, )
        target_mask = (target_sequence != k_stoi[PADDING_TOKEN]).float() # (BLOCK_SIZE, )

        return source_sequence, target_sequence, source_mask, target_mask


def masked_loss(outputs, targets, target_padding_masks):
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    # outputs.shape =  (batch_size, block_size, vocab_size)
    # targets.shape = (batch_size, block_size) targets should be integers
    # target_padding_mask = (batch_size, block_size)
    loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1)) # (batch_size * block_size)
    loss = loss.view(targets.size()) # (batch_size, block_size)
    # Resetting the loss effect of the PADDING_TOKEN token positions as they shouldn't contribute to the loss 
    loss = loss * target_padding_masks # (batch_size, block_size)
    return loss.mean()




def train(model, train_dataloader, val_dataloader, optimizer):
    writer = SummaryWriter(log_dir='logs')
# def train(train_dataloader, val_dataloader):
    pbar = tqdm(total=len(train_dataloader) * NUM_EPOCHS + len(val_dataloader) * NUM_EPOCHS, dynamic_ncols=True)
    # print()
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        for batch_idx, (source_seqs, target_seqs, source_padding_masks, target_padding_masks) in enumerate(train_dataloader):
            source_seqs, target_seqs = source_seqs.to(DEVICE), target_seqs.to(DEVICE)
            source_padding_masks, target_padding_masks = source_padding_masks.to(DEVICE), target_padding_masks.to(DEVICE)

            # # Zeroing the gradients 
            optimizer.zero_grad()
            '''DEBUGGING'''
            # B, T = target_seqs.shape
            # outputs =  torch.rand((B, T, VOCAB_SIZE_K), dtype=torch.float).to(DEVICE)  # Correctly shape the outputs
            # Forward Pass
            outputs = model(source_seqs, target_seqs, src_pad_mask=source_padding_masks, tgt_pad_mask=target_padding_masks)

            # Compute loss
            loss = masked_loss(outputs, target_seqs, target_padding_masks)

            # Backward pass and optimization
            loss.backward()


            # print the training statistics
            if batch_idx % LOG_INTERVAL == 0:
                pbar.update(LOG_INTERVAL)        
                writer.add_scalar('Loss/train', loss.item(), epoch * len(train_dataloader) + batch_idx)        
                # print(f'Epoch: {epoch} [{batch_idx * len(source_seqs)}/{len(train_dataloader.dataset)} \
                #       ({100. * batch_idx / len(train_dataloader):.0f}%)]\tLoss: {loss.item():.6f}\n')
                
            train_loss += loss.item()
        train_loss /= len(train_dataloader)
        # print(f"\nEpoch {epoch}, Training Loss: {train_loss}")
        model.eval()
        val_loss = 0
        # print("\nValidating...")
        with torch.no_grad():
            for batch_idx, (source_seqs, target_seqs, source_padding_masks, target_padding_masks) in enumerate(val_dataloader):
            # Move tensors to the specified device
                source_seqs, target_seqs = source_seqs.to(DEVICE), target_seqs.to(DEVICE)
                source_padding_masks, target_padding_masks = source_padding_masks.to(DEVICE), target_padding_masks.to(DEVICE) 
                
                # # # Forward pass through the model
                # # B, T = target_seqs.shape
                # outputs = torch.rand(B, T, VOCAB_SIZE_K).to(DEVICE)  # Correctly shape the outputs
                outputs = model(source_seqs, target_seqs, src_pad_mask=source_padding_masks, tgt_pad_mask=target_padding_masks)
                # Compute the masked loss
                loss = masked_loss(outputs, target_seqs, target_padding_masks)    
                val_loss += loss.item()

                    # Example of monitoring gradients for attention weights
                for name, param in model.named_parameters():
                    if 'key' in name or 'query' in name or 'value' in name:
                        if param.grad is not None:
                            print(f"Gradients for {name}:")
                            print(param.grad)

        val_loss /= len(val_dataloader)
        pbar.update(len(val_dataloader))

         # Log the epoch training and validation loss
        print(f"\nEpoch: {epoch}\t Training Loss: {train_loss:.6f}\t Validation Loss: {val_loss:.6f}") 
        writer.add_scalar('Loss/train_epoch', train_loss, epoch)
        writer.add_scalar('Loss/val_epoch', val_loss, epoch)
                       
                
    pbar.close()
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Training completed in {total_time:.2f} seconds.")

    writer.close()




def verify_data_loader(dataloader, model):
    model.eval()
    with torch.no_grad():
        for batch_idx, (source_seqs, target_seqs, source_padding_masks, target_padding_masks) in enumerate(dataloader):
            source_seqs, target_seqs = source_seqs.to(DEVICE), target_seqs.to(DEVICE)
            source_padding_masks, target_padding_masks = source_padding_masks.to(DEVICE), target_padding_masks.to(DEVICE)
            outputs = model(source_seqs, target_seqs, src_pad_mask=source_padding_masks, tgt_pad_mask=target_padding_masks)
            print(f"Batch {batch_idx + 1}: Passed through model")
            if batch_idx >= 2:  # Verifying a few batches
                break





if TRAIN:
    english_train, english_val, korean_train, korean_val = train_test_split(
        english_tokens.tolist(), 
        korean_tokens.tolist(), 
        test_size=0.2, 
        random_state=42)
    english_train, english_val, korean_train, korean_val = torch.tensor(english_train),\
          torch.tensor(english_val), torch.tensor(korean_train), torch.tensor(korean_val)
    train_dataset = TextDataset(english_train, korean_train, BLOCK_SIZE)
    val_dataset = TextDataset(english_val, korean_val, BLOCK_SIZE)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # model = Transformer() # Loading the model
    model = Transformer().to(DEVICE) # Loading the model
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # # Verify data loader
    # print("Verifying train dataloader...")
    # verify_data_loader(train_dataloader, model)
    # print("Verifying validation dataloader...")
    # verify_data_loader(val_dataloader, model)


    train(model, train_dataloader, val_dataloader, optimizer)
    # train(train_dataloader, val_dataloader)
    # save the model once training is completed 
    torch.save(model.state_dict(), 'English_to_korean.pth')
    # Example of monitoring gradients for attention weights
    # for name, param in model.named_parameters():
    #     if 'key' in name or 'query' in name or 'value' in name:
    #         if param.grad is not None:
    #             print(f"Gradients for {name}:")
    #             print(param.grad)
    optimizer.step() # updating the parameters
    print("Model English_to_korean.pth saved successfuly!")


DECODE = False

if DECODE:
    pass
