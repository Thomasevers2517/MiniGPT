import sys
sys.path.insert(0, '/Users/rodrigoalvarezlucendo/Desktop/MiniGPT')

from src.tokenizer.openai import OpenAITokenizer
from rbloom import Bloom

# read file.
filename = 'input.txt'
with open(filename, 'r', encoding='utf-8') as f:
    text = f.read()

# tokenize the text
tokenizer = OpenAITokenizer("gpt2")
tokens = tokenizer.encode(text)

# extract n-grams from tokens
def extract_ngrams(tokens, n):
    ngrams = [ tokens[i:i+n] for i in range(len(tokens)-n+1) ]
    return ngrams

ngrams = extract_ngrams(tokens, 3)

# create a bloom filter with 1% false positive rate
bf = Bloom(expected_items=len(ngrams), false_positive_rate=0.01)

for ngram in ngrams:
    bf.add(tokenizer.decode(ngram))

res = tokenizer.decode(ngrams[4]) in bf
print(res)
