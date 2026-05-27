from helpers import get_text
import tiktoken

text = get_text()


vocab = sorted(list(set(text)))
vocab_size = len(vocab)

# Simple Encoder - next step implement the different options in HF
stoi = { ch:i for i,ch in enumerate(vocab)}
itos = { i:ch for i,ch in enumerate(vocab)}


data = tokenize(text)

n = int(0.9*len(data))

training_data = data[:n]
validation_data = data[n:]

block_size = 8

given = training_data[:block_size]
next = training_data[1:block_size+1]

for i in range(block_size):
    x = given[:i+1]
    y = next[i]
    print(f"Given {x} the next character should be {y}")


    