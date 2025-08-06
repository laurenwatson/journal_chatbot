import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

''' Code here taken from Andrej Karpathy's tutorial to build NanoGPT'''

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table=nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        logits=self.token_embedding_table(idx)

        # we get something that is BxTxC - batches x block_size x channels (vocab_size)
        # Pytorch expects BxCxT
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            # stretch out the array to stack up the BxT part
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


def load_data():
    with open('tinyshakespeare/input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    return text

def load_tokeniser(chars):
    ' a very simple tokeniser that converts characters to integers'

    string_to_int = { ch:i for i,ch in enumerate(chars) }
    int_to_string ={ i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [string_to_int[c] for c in s ]
    decode = lambda l: ''.join([int_to_string[i] for i in l])

    return encode, decode

def get_batch(split):
    data = train_data if split == 'train' else val_data
    # generates batch_size number of random ints from len(data)-block_size... so here 4 random start positions in data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # stack combines a sequence of tensors along a new dimension (here stacks them up in rows)
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

if __name__=='__main__':

    # load the data and get vocab
    text = load_data()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    # load tokeniser
    encode, decode = load_tokeniser(chars)

    # tokenise the data
    data = torch.tensor(encode(text), dtype=torch.long)

    # train, test split
    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]

    block_size = 8
    # here block size refers to text block size (how many characters) - the time element
    # we are predicting the next character, character y_i here follows character x_i in the training data
    x = train_data[:block_size]
    y = train_data[1:block_size+1]

    for t in range(block_size):
        context = x[:t+1]
        target = y[t]

    # now we move on to minibatching
    torch.manual_seed(1337)

    batch_size = 4
    block_size = 8

    xb, yb = get_batch('train')
    # xb is now a 4x8 tensor, 4 rows of a length 8 array of character numbers
    # yb is the same but shifted one to the right

    # these 4 x 8 arrays contain a total of 32 training points,
    # this is  as we move from 1 character to full length in each array
    # creating examples of different input length for the transformer

    # for b in range(batch_size):
    #     for t in range(block_size):
    #         context = xb[b, :t+1]
    #         target = yb[b,t]
    #         print(f"when input is {context.tolist()} the target: {target}")
    #

    m = BigramLanguageModel(vocab_size)
    logits, loss = m(xb, yb)
    # size 4x8x65
    # the logits output are for the 4x8 examples, we see a score for which of the 65 vocab items is next
    print(logits.shape)
    print(loss)

    # a little 1x1 tensor holding a 0 to start the generation, 0 is a token is newline character
    idx = torch.zeros((1,1), dtype=torch.long)

    print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))

    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

    batch_size = 32

    for steps in range(5000):
        xb, yb = get_batch('train')

        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(loss.item())

    idx = torch.zeros((1,1), dtype=torch.long)
    print(idx, idx.shape)

    print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))
