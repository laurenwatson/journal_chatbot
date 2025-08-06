from bigram import *

''' Code here taken from Andrej Karpathy's tutorial to build NanoGPT'''


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

def get_batch(split, train_data, val_data, block_size, batch_size):
    data = train_data if split == 'train' else val_data
    # generates batch_size number of random ints from len(data)-block_size... so here 4 random start positions in data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # stack combines a sequence of tensors along a new dimension (here stacks them up in rows)
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

def train(data, encode, decode):

    # train, test split
    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]

    block_size = 8
    x = train_data[:block_size]
    y = train_data[1:block_size+1]

    for t in range(block_size):
        context = x[:t+1]
        target = y[t]

    # now we move on to minibatching
    torch.manual_seed(1337)

    batch_size = 4
    block_size = 8

    xb, yb = get_batch('train', train_data, val_data, block_size, batch_size)

    m = BigramLanguageModel(vocab_size)
    logits, loss = m(xb, yb)
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

    batch_size = 32
    for steps in range(5000):
        xb, yb = get_batch('train',train_data, val_data, block_size, batch_size)

        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return m

if __name__=='__main__':

    # load the data and get vocab
    text = load_data()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    # load tokeniser
    encode, decode = load_tokeniser(chars)

    data = torch.tensor(encode(text), dtype=torch.long)

    m = train(data, encode, decode)

    while True:
        user_input = input("Lauren: ")
        if user_input.lower() in ["quit", "end", "exit", "bye"]:
            print("Journal: Goodbye!")
            break

        encoded_input = torch.tensor([encode(user_input)])
        print(encoded_input)

        # response = respond(user_input)
        response = decode(m.generate(encoded_input, max_new_tokens=30)[0].tolist())

        print("Journal: ", response)
