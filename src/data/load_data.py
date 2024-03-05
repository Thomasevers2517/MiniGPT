def load_data():
    with open('training_data.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    return text