from src.data.load_data import load_data
from src.Embedder import Embedder
import torch
from src.data.data_split import data_split
from src.data.get_batch import get_batch
from src.models.LLM import LLM

def main():
    text = load_data()
    SimpleEmbedder = Embedder('simple', text)
    encoded = SimpleEmbedder.encode(text)
    train_data, val_data = data_split(encoded, 0.8, 'cut')
    xb, yb = get_batch(train_data, 10, 2)
    
    m = LLM("simple")(SimpleEmbedder.vocab_size)
    
    logits, loss = m(xb, yb)
    print(logits.shape)
    print(loss)

    print(SimpleEmbedder.decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))
    


if __name__ == "__main__":
    main()