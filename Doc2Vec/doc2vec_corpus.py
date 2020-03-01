import glob
import MeCab
import nltk
import numpy as np
import pickle
import re

from collections import Counter

with open("../Word2Vec/stop_words.pkl", "rb") as f:
    stop_words = pickle.load(f)

jp_sent_tokenizer = nltk.RegexpTokenizer("[^　「」！？。]*[！？。]")
mecab = MeCab.Tagger("mecabrc")

UNK = "<UNK>"  # unknown


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
    word_counts = [x for x in counter.items() if x[1] >= 2]  # less 2 count word is not common word
    word_counts.sort(key=lambda x: x[1], reverse=True)
    print("Number of words:", len(word_counts) + 1)  # plus 1 is <UNK>
    word_list = [x[0] for x in word_counts]

    word_list.append(UNK)
    word2id = dict([(x, y) for (y, x) in enumerate(word_list)])
    id2word = dict([(x, y) for (x, y) in enumerate(word_list)])

    return word2id, id2word, counter


if __name__ == "__main__":
    """ livedoor NEWS Corpus """
    corpus = []
    document_train = []
    document_val = []

    dir_list = glob.glob("./text/*")
    dir_list = [dir_name for dir_name in dir_list if not dir_name.endswith("txt")]
    for i, dir_file in enumerate(dir_list):
        file_list = glob.glob(dir_file + "/*.txt")
        file_list = [file for file in file_list if not "LICENSE" in file]  # remove LICENSE.txt

        #
        # training data
        #
        for file in file_list[:-10]:
            with open(file, encoding="UTF-8") as f:
                text = f.readlines()

            text_list = text_cleaning(text)  # text cleaning
            sent_list = sentence_tokenized(text_list)  # sentence tokenized
            word_list = word_tokenized(sent_list)  # word tokenized

            token_list = []
            for words in word_list:
                for token in words:
                    token_list.append(token)
            token_list.append(i)

            corpus.append(token_list[:-1])
            document_train.append(token_list)

        #
        # validation data
        #
        for file in file_list[-10:]:
            with open(file, encoding="UTF-8") as f:
                text = f.readlines()

            text_list = text_cleaning(text)  # text cleaning
            sent_list = sentence_tokenized(text_list)  # sentence tokenized
            word_list = word_tokenized(sent_list)  # word tokenized

            token_list = []
            for words in word_list:
                for token in words:
                    token_list.append(token)
            token_list.append(i)

            document_val.append(token_list)

    #
    # word2id and id2word dictionary
    #
    word2id, id2word, _ = create_word2id(corpus)
    
    with open("word2id.pkl", "wb") as f:
        pickle.dump(word2id, f)
    print("\nSaved word2id.")

    with open("id2word.pkl", "wb") as f:
        pickle.dump(id2word, f)
    print("Saved id2word.\n")

    #
    # document2id
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
    # word and label
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

    print("\nNumber of validation samples", num_samples)
    
