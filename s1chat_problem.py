import re
import os

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.models import transformer
from tensor2tensor.utils import registry

@registry.register_problem("s1_chat")
class S1Chat(text_problems.Text2TextProblem):
    @property
    def vocab_filename(self):
        return "vocab.txt"

    @property
    def oov_token(self):
        return "$$"
   
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
        title_file = open(os.path.join(data_dir, "title.txt"))
        reply_file = open(os.path.join(data_dir, "reply.txt"))
        for title in title_file:
            reply = reply_file.readline()
            yield {
                "inputs": title.strip(),
                "targets": reply.strip()
            }