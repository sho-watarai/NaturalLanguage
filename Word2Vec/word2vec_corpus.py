import MeCab
import numpy as np
import pickle
import re

from collections import Counter

neologd = "C:/Users/user/AppData/Local/Packages/KaliLinux.54290C8133FEE_ey8k8hqnwqnmg/LocalState" \
          "/rootfs/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd"

NUM = "N"  # number
UNK = "<UNK>"  # unknown

with open("./stop_words.pkl", "rb") as f:
    stop_words = pickle.load(f)

num_window = 5


class MeCabTokenizer:
    def __init__(self):
        self.tokenizer = MeCab.Tagger("-O wakati -d %s" % neologd)

    def tokenize(self, string):
        #
        # noun, verb, and adjectives
        #
        node = self.tokenizer.parseToNode(string)
        word = []
        while node:
            if node.feature.startswith("名詞"):
                word.append(node.surface)
                node = node.next
            elif node.feature.startswith("動詞"):
                word.append(node.surface)
                node = node.next
            elif node.feature.startswith("形容詞"):
                word.append(node.surface)
                node = node.next
            else:
                node = node.next

        return word


def text_cleaning(s):
    #
    # specific
    #
    s = re.sub(r"\u3000", "", s)  # remove indent
    s = re.sub(r"\(.+\)", "", s)  # remove ruby

    #
    # normalization
    #
    s = re.sub("[˗֊‐‑‒–⁃⁻₋−]+", "-", s)  # normalize hyphens
    s = re.sub("[﹣－ｰ—―─━ー]+", "ー", s)  # normalize choonpus
    s = re.sub("[~∼∾〜〰～]", "", s)  # remove tildes

    s = s.lower()  # normalize alphabet to lowercase
    s = s.translate({ord(x): ord(y) for x, y in zip(  # normalize half-width symbols to full-width symbols
        "!\"#$%&'()*+,-./:;<=>?@[¥]^_`{|}~｡､･｢｣",
        "！”＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［￥］＾＿｀｛｜｝〜。、・「」")})

    #
    # reduce redundancy
    #
    s = re.sub(r"！+", "！", s)
    s = re.sub(r"？？+", "？", s)
    s = re.sub(r"…+", "…", s)
    s = re.sub(r"w+w", "。", s)

    s = re.sub(r"[^ a-z0-9ぁ-んァ-ン一-龥ー、。！？]", "", s)

    return s


def create_word2id(tokenized):
    counter = Counter()
    for t in tokenized:
        counter.update(t)

    print("Number of total words:", len(counter))
    word_counts = [x for x in counter.items() if x[1] >= 2]  # less 2 count word is not common word
    word_counts.sort(key=lambda x: x[1], reverse=True)
    print("Number of words:", len(word_counts) + 1)  # plus 1 is <UNK>
    word_list = [x[0] for x in word_counts]

    word_list.append(UNK)
    word2id = dict([(x, y) for (y, x) in enumerate(word_list)])
    id2word = dict([(x, y) for (x, y) in enumerate(word_list)])

    return word2id, id2word


def unigram_distribution(tokenized):
    counter = Counter()
    for t in tokenized:
        counter.update(t)

    sampling_weights = np.zeros((len(counter),), dtype="float32")
    for key, value in counter.items():
        sampling_weights[key] = value

    sampling_weights /= sampling_weights.sum()

    np.save("./sampling_weights.npy", sampling_weights)
    print("Saved unigram distribution as sampling_weights.npy\n")

    return sampling_weights


if __name__ == "__main__":
    #
    # MeCab with NEologd
    #
    mecab = MeCabTokenizer()

    #
    # preprocessing
    #
    with open("./MagicalChildren.txt", encoding="utf-8") as f:
        text = f.readlines()

    tokenized_list = []
    for t in text:
        t = text_cleaning(t)  # text cleaning
        
        word_list = mecab.tokenize(t)  # word tokenized

        word_list = [re.sub(r"[0-9]+|[0-9].+[0-9]", NUM, word) for word in word_list]  # word normalization

        for w in word_list:
            if w in stop_words:
                word_list.remove(w)  # remove stop words

        tokenized_list.append(word_list)

    #
    # word2id and id2word dictionary
    #
    word2id, id2word = create_word2id(tokenized_list)

    with open("./word2id.pkl", "wb") as f:
        pickle.dump(word2id, f)
    print("\nSaved word2id.")

    with open("./id2word.pkl", "wb") as f:
        pickle.dump(id2word, f)
    print("Saved id2word.\n")

    #
    # unigram distribution
    #
    tokenized_list_replace = []
    for words in tokenized_list:
        tokenized_list_replace.append([word2id[word] if word in word2id else word2id[UNK] for word in words])

    sampling_weights = unigram_distribution(tokenized_list_replace)

    #
    # create corpus
    #
    corpus = []
    for words in tokenized_list_replace:
        for word in words:
            corpus.append(word)

    corpus = np.array(corpus, dtype=int)

    #
    # Skip-gram
    #
    words = corpus[num_window:-num_window]
    targets = []
    for i in range(num_window, len(corpus) - num_window):
        word_list = []
        for j in range(-num_window, num_window + 1):
            if j == 0:
                continue
            word_list.append(corpus[i + j])
        targets.append(word_list)

    print("Skip-gram\n")

    num_samples = 0
    with open("./skipgram_corpus.txt", "w") as word_file:
        for i in range(len(words)):
            for j in range(num_window * 2):
                word_file.write("|word {}:1\t|target {}:1\n".format(words[i], targets[i][j]))

                num_samples += 1
                if num_samples % 10000 == 0:
                    print("Now %d samples..." % num_samples)

    print("\nNumber of samples", num_samples)

    #
    # CBOW
    #
    targets = corpus[num_window:-num_window]
    words = []
    for i in range(num_window, len(corpus) - num_window):
        word_list = []
        for j in range(-num_window, num_window + 1):
            if j == 0:
                continue
            word_list.append(corpus[i + j])
        words.append(word_list)

    print("\nCBOW\n")

    num_samples = 0
    with open("./cbow_corpus.txt", "w") as word_file:
        for i in range(len(words)):
            word_file.write("{} |word {}:1\t|target {}:1\n".format(i, words[i][0], targets[i]))
            for j in range(1, num_window * 2):
                word_file.write("{} |word {}:1\n".format(i, words[i][j]))

            num_samples += 1
            if num_samples % 10000 == 0:
                print("Now %d samples..." % num_samples)

    print("\nNumber of samples", num_samples)
    
