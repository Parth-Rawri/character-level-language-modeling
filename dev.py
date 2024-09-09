
# Establishing baseline: Bigram model
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(1336)

# hyperparameters
batch_size = 4 # number of independent sequences we process in parallel
block_size = 8 # maximum context length for predictions
max_iters = 10000
learning_rate = 1e-3
eval_iterval = 300
eval_iters = 200
n_embd = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

# generating a random batch of data
def get_batch(split):
    data = train_data if split=='train' else val_data
    ix = torch.randint(len(data)-block_size, (batch_size, ))
    xb = torch.stack([data[x:x+block_size] for x in ix])
    yb = torch.stack([data[x+1:x+block_size+1] for x in ix])
    xb, yb = xb.to(device), yb.to(device)
    return xb, yb

# estimate a less noisy version of loss
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'eval']:
        losses = torch.zeros(eval_iters)
        for iter in range(eval_iters):
            xb, yb = get_batch(split)
            logits, loss = model(xb, yb)
            losses[iter] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Bigram(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        """
        idx [B,T], targets dimension [B,T]
        logits dimentsion [B,T,C] where, B is batch, T is Time, C is the embedding dimension
        CE expects:
            Input (logits) to be of shape [N, C], where N is the number of samples and C is the number of classes.
            The target should be of shape [N], where each value is the target class for the corresponding sample.
        O/P logits [B*T, C] OR [B,T,C], loss
        """
        token_embd = self.token_embedding_table(idx) # (B,T,embedding_dim)
        logits = self.lm_head(token_embd) # (B,T,vocab_size)

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

model = Bigram()
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_iterval == 0:
        losses = estimate_loss()
        print(f'step {iter}, train loss {losses['train']:.4f}, eval loss {losses['eval']:.4f}')

    # sample a batch of data
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    # evaluate the loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=300)[0].tolist()))
