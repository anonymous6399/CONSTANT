
vocab = {
    'base': 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
}

class Char_Tokenizer:
    def __init__(self, max_length=10, padding=False, truncating=False) -> None:
        self.vocab = vocab['base']
        self.str2idx = {char:idx for idx, char in enumerate(self.vocab)}
        self.idx2str = {v:k for k, v in self.str2idx.items()}
        self.special_chars = {'PAD_TOKEN': len(self.vocab)}
        self.vocab_size = len(self.vocab) + len(self.special_chars.keys())
        self.batch_max_length = max_length
        self.padding = padding
        self.truncating = truncating
        
    def encode(self, text):       
        text = [self.str2idx[char] for char in text]
        if self.batch_max_length is not None:
            if len(text) > self.batch_max_length and self.truncating:
                text = text[:self.batch_max_length]
            pad_len = self.batch_max_length - len(text)
            if pad_len > 0 and self.padding:
                text = text + [self.special_chars['PAD_TOKEN']]*pad_len
        return text
        