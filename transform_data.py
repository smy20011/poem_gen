import json
import itertools
import functools
import collections
import re
import tensorflow as tf
import jieba
import multiprocessing

VOCAB_SIZE = 5000

def read_from_json():
    f = open("dump.json")
    for line in f:
        line = line.strip()
        while line and not line.endswith("}"):
            line = line[:-1]
        while line and not line.startswith("{"):
            line = line[1:]
        try:
            thread = json.loads(line)
            title = thread["title"]
            for post in thread["posts"][1:]:
                if "content" in post and "发表于" not in post["content"]:
                    content = post["content"]
                    for reply_line in re.split(r"\r|\n", content):
                        if len(reply_line) > 2:
                            yield (title, reply_line)
        except Exception:
            pass

def tokenlize_input(data: str):
    data = data.replace(" ", "")
    return list(jieba.cut(data))

def tokenlize(data):
    title, reply = data
    return tokenlize_input(title), tokenlize_input(reply)

def chunk(generator, size):
    while True:
        item = list(itertools.islice(generator, size))
        if item:
            yield item
        else:
            break

def generate_vocab(generator):
    counter = collections.Counter()
    last_title = ""
    for title, reply in generator:
        tokens = []
        if title != last_title:
            last_title = title
            tokens = title
        tokens = itertools.chain(tokens, reply)
        for token in tokens:
            counter[token] += 1
    return counter

def combine_counters(counters):
    counter = collections.Counter()
    for c in counters:
        counter.update(c)
    return counter

def save_to_file(corpus, vocab):
    title = open("title.txt", "w")
    reply = open("reply.txt", "w")
    vocab_file = open("vocab.txt", "w")
    for t, r in corpus:
        print(" ".join(t), file=title)
        print(" ".join(r), file=reply)
    
    for word in vocab.most_common(VOCAB_SIZE):
        print(word[0], file=vocab_file)
    print("$$", file=vocab_file)

    title.close()
    reply.close()
    vocab_file.close()
    
pool = multiprocessing.pool.Pool()
data = read_from_json()
corpus = pool.map(tokenlize, data)
corpus, vocab = itertools.tee(corpus)
vocab = chunk(vocab, 10000)
vocab = pool.map(generate_vocab, vocab)
vocab = combine_counters(vocab)
save_to_file(corpus, vocab)