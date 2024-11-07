# add all  your Encoder and Decoder code here
import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F

batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 6  # Number of transformer layers


d_model = n_embd

eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training

# # # ENCODER # # #
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layer, block_size, n_hidden=100, dropout=0.1):
        super().__init__()

        #Created token and pos embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)

        #initialized the encoder blocks
        self.blocks = nn.ModuleList([EncoderBlock(d_model, n_head, n_hidden) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, idx):
        B, T = idx.shape
        token_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = token_emb + pos_emb
        
        #In the attention weights, I grabbed them from the blocks and put them in a list
        attention_weights = []
        for block in self.blocks:
            x, attn = block(x)
            attention_weights.append(attn)
        
        x = self.ln_f(x)
        return x, attention_weights


class Head(nn.Module):
    def __init__(self, head_dim, d_model):
        super().__init__()
        self.head_dim = head_dim
        self.key = nn.Linear(d_model, head_dim, bias=False)
        self.query = nn.Linear(d_model, head_dim, bias=False)
        self.value = nn.Linear(d_model, head_dim, bias=False)
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  
        q = self.query(x)
        v = self.value(x)

        # Compute attention scores
        attn_scores = q @ k.transpose(-2, -1) * (self.head_dim ** -0.5) # (B, T, T)
        attn_weights = F.softmax(attn_scores, dim=1)

        # Compute output
        output = attn_weights @ v # (B, T, head_dim)
        
        return output, attn_weights


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_head, n_hidden):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, d_model)
        self.ffwd = FeedForward(d_model, n_hidden)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x_ln1 = self.ln1(x)
        sa_out, attn = self.sa(x_ln1)
        x = x + sa_out
        x = x + self.ffwd(self.ln2(x))
        return x, attn
    

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.heads = nn.ModuleList([Head(self.head_dim, d_model) for _ in range(n_head)])
        self.proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        B, T, C = x.shape
        outputs = []
        attentions = []
        for head in self.heads:
            output, attn = head(x)
            outputs.append(output)
            attentions.append(attn)
        concat_outputs = torch.cat(outputs, dim=-1)
        output = self.proj(concat_outputs)
        attentions = torch.stack(attentions, dim=1)
        return output, attentions
        

class FeedForward(nn.Module):
    def __init__(self, n_embd, n_hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_embd)
        )

    def forward(self, x):
        return self.net(x)

#Combined function that takes the encoder and the classifier
class EncoderClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layer, block_size, hidden):
        super().__init__()
        self.encoder = Encoder(vocab_size, n_embd, n_head, n_layer, block_size)
        self.classifier = FeedForward(d_model, hidden)

    def forward(self, x):
        encoded, weights = self.encoder(x)
        encoded_mean = encoded.mean(dim=1)
        return self.classifier(encoded_mean), weights

# # # DECODER # # #


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layer, n_head):
        super().__init__()
        
        # Embedding layers for tokens and positions
        self.token_embedding = nn.Embedding(vocab_size,d_model)

        self.vocabSize = vocab_size
                
        # Stack of decoder layers
        self.layers = nn.ModuleList([DecoderBlock(d_model, n_head) for _ in range(n_layer)])
        

        self.ln_n = nn.LayerNorm(d_model)
        self.ret = FeedForwardDecoder(d_model, 100, vocab_size)


    def forward(self, X, Y=None):
        
        B, T = X.shape
        
        # Embeddings for tokens and positions
        token_emb = self.token_embedding(X)   # Shape: (B, T, d_model)
        
        
        x = token_emb

        attention_weights = []
        
        for layer in self.layers:
            x , attn = layer(x)
            attention_weights.append(attn)

        # Project logits to vocab size
        x = self.ln_n(x)
        logits = self.ret(x)

        if Y is not None:
            # Compute cross-entropy loss if targets are provided

            logits_flattened = logits.view(-1, self.vocabSize)
            Y = Y.view(-1)

            # Two different types of loss functions, Left this here as a backup
            # # loss_function = nn.CrossEntropyLoss()
            # # loss = loss_function(logits_flattened, Y)

            return F.cross_entropy(logits_flattened, Y), attention_weights
                    
        return logits, attention_weights
    
class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        
        #Same as before but with new Feed Forward with Dropout
        self.masked_attn = MaskedMultiHeadAttention(n_head=n_head, d_model=d_model)
        self.ffwd = FeedForwardDecoder(d_model, 100, d_model)
        
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Apply layer norm before attention
        attn_out, attn_weights = self.masked_attn(self.ln1(x))
        x = x + attn_out
        
        # Apply layer norm before feedforward
        x_norm = self.ln2(x)
        x = x + self.ffwd(x_norm)
        
        return x, attn_weights

# NEW FEEDFORWARD with DROPOUT  layers. Set Dropout to 0.2
class FeedForwardDecoder(nn.Module):
    def __init__(self, d_model, n_hidden, ret):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, n_hidden),
            nn.ReLU(),
            #Dropout 20%
            nn.Dropout(0.2),
            nn.Linear(n_hidden, ret)
        )

    def forward(self, x):
        return self.net(x)

#Created masked head so it is different than encoder
class MaskedHead(nn.Module):
    def __init__(self, head_dim, d_model):
        super().__init__()
        self.head_dim = head_dim
        #KQV matrices
        self.key = nn.Linear(d_model, head_dim, bias=False)
        self.query = nn.Linear(d_model, head_dim, bias=False)
        self.value = nn.Linear(d_model, head_dim, bias=False)
    
    def forward(self, x, mask):
        B, T, C = x.shape

        k = self.key(x)  
        q = self.query(x)
        v = self.value(x)

        # Compute attention scores
        attn_scores = q @ k.transpose(-2, -1) * (C ** -0.5)
        
        # Apply mask if set to 1
        if mask is not None:
            #Generates 1s for the bottom triangle and fills -inf on top triangle
            tril = torch.tril(torch.ones(T,T, device = x.device))
            attn_scores = attn_scores.masked_fill(tril == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=1)

        # Compute output
        output = attn_weights @ v
        
        return output, attn_weights


#Created masked multihead so it is different than encoder
class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model):
        super().__init__()
        #Created a new masked head instead of regular to use for decoder
        self.heads = nn.ModuleList([MaskedHead(d_model // n_head, d_model) for _ in range(n_head)])
        self.proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        outputs = []
        attentions = []
        for head in self.heads:
            output, attn = head(x, True)
            outputs.append(output)
            attentions.append(attn)
        concat_outputs = torch.cat(outputs, dim=-1)
        output = self.proj(concat_outputs)
        attentions = torch.stack(attentions, dim=1)
        return output, attentions