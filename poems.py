import re
import sys
import os
import collections
import jieba

from typing import Iterable
Poem = collections.namedtuple("Poem", "title, verses")

def get_all_poems():
    f = open("poems.txt")
    for line in f:
        line = line.strip()
        line = re.sub("（.*?）", "", line)
        title, *verses = re.split("[:，。？]", line)
        verses = list(filter(lambda s : len(s) > 0, verses))
        yield Poem(title, verses)

def tokenlize(s):
    return list(s)

def tokenlize_poem(poems: Iterable[Poem]):
    for poem in poems:
        new_title = tokenlize(poem.title)
        new_verses = [tokenlize(verse) for verse in poem.verses]
        yield Poem(new_title, new_verses)

def generate_vocab(poems: Iterable[Poem]):
    counter = collections.Counter()
    for poem in poems:
        all_str = poem.verses + [poem.title]
        for s in all_str:
            for word in s:
                counter[word] += 1
    return list(counter.keys())

def is_tang_poem(poem: Poem):
    return all(map(lambda v: len(v) == len(poem.verses[0]), poem.verses)) and (len(poem) % 2 == 0)

def save_vocab_to_file(vocab, filename):
    f = open(filename, "w")
    for word in vocab:
        print(word, file = f)
    f.close()

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

@registry.register_problem("chinese_poem")
class ChinesePoem(text_problems.Text2TextProblem):
    TITLE_SPLITTER = "$$"
    def __init__(self, *args):
        super(ChinesePoem, self).__init__(*args)
        self.poems = list(get_all_poems())
        print("All Poems Count: ", len(self.poems))
        self.poems = list(filter(is_tang_poem, self.poems))
        print("Tang Poems Count: ", len(self.poems))
        self.poems = list(tokenlize_poem(self.poems))
        vocab = generate_vocab(self.poems)
        vocab = vocab + [ChinesePoem.TITLE_SPLITTER]
        save_vocab_to_file(vocab, os.path.join("artifacts", self.vocab_filename))

    @property
    def oov_token(self):
        return "$$$"
   
    @property
    def is_generate_per_split(self):
        return False
    
    @property
    def vocab_type(self):
        return text_problems.VocabType.TOKEN

    @property
    def dataset_splits(self):
        return [
            {"split": problem.DatasetSplit.TRAIN, "shards": 9},
            {"split": problem.DatasetSplit.EVAL, "shards": 1},
        ]
    
    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del data_dir
        del tmp_dir
        del dataset_split

        for poem in self.poems:
            for i in range(len(poem.verses) // 2):
                yield {
                    "inputs": " ".join(poem.verses[i * 2]),
                    "targets": " ".join(poem.verses[i * 2 + 1])
                }