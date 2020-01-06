import MeCab
import nltk
import numpy as np
import pickle
import re

from collections import Counter

with open("./stop_words.pkl", "rb") as f:
    stop_words = pickle.load(f)

num_window = 5
jp_sent_tokenizer = nltk.RegexpTokenizer("[^　「」！？。]*[！？。]")
mecab = MeCab.Tagger("mecabrc")


def mecab_tokenizer(text):
    node = mecab.parseToNode(text)
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


def text_cleaning(text):
    text_list = []
    for t in text:
        t = re.sub("\u3000|\n|、|…", "", t)
        if t == "":
            continue
        else:
            text_list.append(t)
    return text_list


def sentence_tokenized(text_list):
    sent_list = []
    for t in text_list:
        sentences = jp_sent_tokenizer.tokenize(t)
        for s in sentences:
            sent_list.append(s)
    return sent_list


def word_tokenized(sent_list):
    tokenized_list = []
    for s in sent_list:
        word_list = []
        tokenized_words = mecab_tokenizer(s)
        for w in tokenized_words:  # remove stop words
            if w in stop_words:
                continue
            else:
                word_list.append(w)
        tokenized_list.append(word_list)
    return tokenized_list


def create_word2id(token_list):
    counter = Counter()
    for t in token_list:
        counter.update(t)

    print("Number of total words:", len(counter))
    word_counts = [x for x in counter.items() if x[1] >= 1]
    word_counts.sort(key=lambda x: x[1], reverse=True)
    word_list = [x[0] for x in word_counts]

    word2id = dict([(x, y) for (y, x) in enumerate(word_list)])
    id2word = dict([(x, y) for (x, y) in enumerate(word_list)])

    return word2id, id2word, counter


def unigram_distribution(counter, word2id):
    sampling_weights = np.zeros((len(counter),), dtype="float32")
    for key, value in counter.items():
        sampling_weights[word2id[key]] = value

    sampling_weights /= sampling_weights.sum()

    np.save("sampling_weights.npy", sampling_weights)
    print("Saved words distribution as sampling_weights.npy\n")

    return sampling_weights


if __name__ == "__main__":
    with open("./MagicalChildren.txt", encoding="UTF-8") as f:
        text = f.readlines()

    text_list = text_cleaning(text)  # text cleaning

    sent_list = sentence_tokenized(text_list)  # sentence tokenized

    word_list = word_tokenized(sent_list)  # word tokenized

    #
    # word2id and id2word dictionary
    #
    word2id, id2word, counter = create_word2id(word_list)

    with open("word2id.pkl", "wb") as f:
        pickle.dump(word2id, f)
    print("\nSaved word2id.")

    with open("id2word.pkl", "wb") as f:
        pickle.dump(id2word, f)
    print("Saved id2word.\n")

    #
    # Unigram word distribution
    #
    sampling_weights = unigram_distribution(counter, word2id)

    #
    # create corpus
    #
    corpus = []
    for words in word_list:
        for w in words:
            corpus.append(w)

    corpus = np.array([word2id[word] for word in corpus], dtype=int)

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

    words, targets = np.array(words, dtype=int), np.array(targets, dtype=int)

    print("Skip-gram")

    num_samples = 0
    with open("./corpus_skipgram.txt", "w") as word_file:
        for i in range(len(words)):
            for j in range(num_window * 2):
                word_file.write("|word {}:1\t|target {}:1\n".format(words[i], targets[i, j]))

                num_samples += 1
                if num_samples % 10000 == 0:
                    print("Now %d samples..." % num_samples)

    print("\nNumber of samples", num_samples)
