import torch

def load_data():
    with open('tinyshakespeare/input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    return text

def load_tokeniser():
    ' a very simple tokeniser that converts characters to integers'

    string_to_int = { ch:i for i,ch in enumerate(chars) }
    int_to_string ={ i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [string_to_int[c] for c in s ]
    decode = lambda l: ''.join([int_to_string[i] for i in l])

    return encode, decode



if __name__=='__main__':

    # load the data and get vocab
    text = load_data()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    # load tokeniser
    encode, decode = load_tokeniser()

    # tokenise the data
    data = torch.tensor(encode(text), dtype=torch.long)

    # train, test split
    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]
