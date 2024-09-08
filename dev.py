
# Establishing baseline: Bigram model
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1337)

# hyperparameters
batch_size = 4 # number of independent sequences we process in parallel
block_size = 8 # maximum context length for predictions
max_iters = 10000
learning_rate = 1e-3

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# extracting vocabulary
characters = sorted(list(set(text)))
vocab_size = len(characters)
print(f'vocab size: {vocab_size}')
print(''.join(characters))

# build text tokenizer
stoi = { c:idx for idx, c in enumerate(characters) }
itos = { idx:c for idx, c in enumerate(characters) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string and output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers and output a string

# encode entire dataset and store as torch.Tensor
data = torch.tensor(encode(text), dtype=torch.long)

# split the dataset
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# generating a batch of data
def get_batch(split):
    data = train_data if split=='train' else val_data
    ix = torch.randint(len(train_data)-block_size, (batch_size, ))
    x = torch.stack([data[x:x+block_size] for x in ix])
    y = torch.stack([data[x+1:x+block_size+1] for x in ix])
    return x, y

class Bigram(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=vocab_size)

    def forward(self, idx, targets=None):
        """
        idx [B,T], targets dimension [B,T]
        logits dimentsion [B,T,C] where, B is batch, T is Time, C is the embedding dimension
        CE expects:
            Input (logits) to be of shape [N, C], where N is the number of samples and C is the number of classes.
            The target should be of shape [N], where each value is the target class for the corresponding sample.
        O/P logits [B*T, C] OR [B,T,C], loss
        """
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits, targets = logits.view(B*T, C), targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        I/P idx dim [B,T], O/P idx dim [B, T+max_new_tokens]
        new tokens are added in the Time dimension for each example in the Batch
        """
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :] # plucking out the logits of the last element in T having dim (B, C)
            probs = F.softmax(logits, dim=-1) # dim (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # dim (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # dim (B, T+1)
        return idx

model = Bigram(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # sample a batch of data
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    # evaluate the loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss)

print(decode(model.generate(torch.zeros((1,1), dtype=torch.long), max_new_tokens=300)[0].tolist()))

context = torch.zeros((1,1), dtype=torch.long)
print(decode(model.generate(context, max_new_tokens=300)[0].tolist()))
