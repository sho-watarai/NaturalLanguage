import glob
import MeCab
import pickle
import re

from collections import Counter

NUM = "N"  # number
UNK = "<UNK>"  # unknown

neologd_path = "C:/Users/user/AppData/Local/Packages/KaliLinux.54290C8133FEE_ey8k8hqnwqnmg/LocalState/rootfs" \
               "/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd"

with open("../Word2Vec/stop_words.pkl", "rb") as f:
    stop_words = pickle.load(f)


class MeCabTokenizer:
    def __init__(self):
        self.tokenizer = MeCab.Tagger("-O wakati -d %s" % neologd_path)

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
    # livedoor specific
    #
    s = re.sub(r"\u3000|\n|\t", "", s)  # remove indent, newline, and tab
    s = re.sub(r"http\S+", "", s)  # remove html
    s = re.sub("[0-9].*", "", s)  # remove time

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

    return s


def document_preprocessing(text, label):
    tokenized_list = []
    for t in text:
        t = text_cleaning(t)  # text cleaning

        if not t:
            continue

        word_list = mecab.tokenize(t)  # word tokenized

        word_list = [re.sub(r"[0-9]+|[0-9].+[0-9]", NUM, word) for word in word_list]  # word normalization

        for w in word_list:
            if w in stop_words:
                word_list.remove(w)  # remove stop words

        for w in word_list:
            tokenized_list.append(w)

    tokenized_list.append(label)  # category label

    return tokenized_list


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


if __name__ == "__main__":
    #
    # MeCab with NEologd
    #
    mecab = MeCabTokenizer()

    corpus = []
    document_train = []
    document_val = []

    dir_list = glob.glob("./text/*")
    dir_list = [dir_name for dir_name in dir_list if not dir_name.endswith("txt")]

    for label, dir_file in enumerate(dir_list):
        file_list = glob.glob(dir_file + "/*.txt")
        file_list = [file for file in file_list if not "LICENSE" in file]  # remove LICENSE.txt

        #
        # training data
        #
        for file in file_list[:-10]:
            with open(file, encoding="utf-8") as f:
                text = f.readlines()

            tokenized_list = document_preprocessing(text, label)

            corpus.append(tokenized_list[:-1])
            document_train.append(tokenized_list)

        #
        # validation data
        #
        for file in file_list[-10:]:
            with open(file, encoding="utf-8") as f:
                text = f.readlines()

            tokenized_list = document_preprocessing(text, label)

            document_val.append(tokenized_list)

    #
    # word2id and id2word dictionary
    #
    word2id, id2word = create_word2id(corpus)

    with open("./word2id.pkl", "wb") as f:
        pickle.dump(word2id, f)
    print("\nSaved word2id.")

    with open("./id2word.pkl", "wb") as f:
        pickle.dump(id2word, f)
    print("Saved id2word.\n")

    #
    # doc2id
    #
    doc_train = []
    for doc in document_train:
        doc2id = [word2id[word] if word in word2id else word2id[UNK] for word in doc[:-1]]
        doc2id.append(doc[-1])
        doc_train.append(doc2id)

    doc_val = []
    for doc in document_val:
        doc2id = [word2id[word] if word in word2id else word2id[UNK] for word in doc[:-1]]
        doc2id.append(doc[-1])
        doc_val.append(doc2id)

    #
    # document and label
    #
    num_samples = 0
    with open("./train_doc2vec_map.txt", "w") as map_file:
        for i, doc in enumerate(doc_train):
            map_file.write("{} |word {}:1\t|label {}:1\n".format(i, doc[0], doc[-1]))
            for j in range(1, len(doc)-1):
                map_file.write("{} |word {}:1\n".format(i, doc[j]))

            num_samples += 1
            if num_samples % 1000 == 0:
                print("Now %d samples..." % num_samples)

    print("\nNumber of training samples", num_samples)

    num_samples = 0
    with open("./val_doc2vec_map.txt", "w") as map_file:
        for i, doc in enumerate(doc_val):
            map_file.write("{} |word {}:1\t|label {}:1\n".format(i, doc[0], doc[-1]))
            for j in range(1, len(doc) - 1):
                map_file.write("{} |word {}:1\n".format(i, doc[j]))

            num_samples += 1

    print("Number of validation samples", num_samples)
    
