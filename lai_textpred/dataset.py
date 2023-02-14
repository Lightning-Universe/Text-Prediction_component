import re
import string
from collections import Counter

from lightning.pytorch.utilities.rank_zero import rank_zero_info
from lightning_gpt.data import CharDataset


class WordDataset(CharDataset):
    def __init__(self, data: str, block_size):
        words = self.remove_punctuation(data).lower().replace("\n", " ").split(" ")
        words.remove("")
        unique_words_counter = Counter(words)

        data_size, vocab_size = len(words), len(unique_words_counter)
        rank_zero_info("data has %d words, %d unique." % (data_size, vocab_size))

        self.stoi = {word: i for i, (word, _) in enumerate(unique_words_counter.most_common())}
        self.itos = {v: k for k, v in self.stoi.items()}
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = words

    @staticmethod
    def remove_punctuation(text: str):
        return re.sub("[%s]" % re.escape(string.punctuation), "", text)
