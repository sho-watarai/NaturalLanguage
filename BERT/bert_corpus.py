import glob
import nltk
import numpy as np
import random
import re
import sentencepiece as spm

jp_sentence = nltk.RegexpTokenizer("[^。]*[。]")

num_word = 32000

CLS = 1  # [CLS]
SEP = 2  # [SEP]
MASK = 3  # [MASK]

spm_path = "./jawiki.model"

rng = random.Random()

masked_rate = 0.15

max_seq_len = 128


class SentencePiece:
    def __init__(self, spm_path):
        self.model = spm.SentencePieceProcessor()
        self.model.Load(spm_path)

    def encode(self, string):
        return self.model.EncodeAsIds(string)

    def encode_pieces(self, string):
        return self.model.encode_as_pieces(string)

    def decode(self, ids):
        return self.model.DecodeIds(ids)

    def decode_pieces(self, pieces):
        return self.model.decode_pieces(pieces)


def text_cleaning(s):
    #
    # corpus specific
    #
    s = s.strip(" ")  # remove beginning and ending space
    s = re.sub(r"。」", "」", s)  # normalize 。」 」
    s = re.sub(r"([あ-んア-ン一-龥ー、。])\s+((?=[あ-んア-ン一-龥ー、。]))", r"\1\2", s)  # remove space between kana-kanji
    s = re.sub(r"<.+>", "", s)  # remove html tag

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
    # remove brackets and string etc.
    #
    s = re.sub(r"[\u3000\n\t\r]", "", s)

    brackets = ["（[^（|^）]*）", "【[^【|^】]*】", "＜[^＜|^＞]*＞", "［[^［|^］]*］", "｛[^｛|^｝]*｝",
                "〔[^〔|^〕]*〕", "〈[^〈|^〉]*〉"]
    while sum([1 if re.search(bracket, s) else 0 for bracket in brackets]):
        for bracket in brackets:
            s = re.sub(bracket, "", s)

    #
    # reduce redundancy
    #
    s = re.sub(r"！+", "！", s)
    s = re.sub(r"？+", "？", s)
    s = re.sub(r"…+", "…", s)
    s = re.sub(r"。+", "。", s)
    s = re.sub(r"、+", "、", s)
    s = re.sub(r"\s+", " ", s)

    return s


def bert_preprocessing():
    #
    # Wikipedia
    #
    file_list = glob.glob("./jawiki/AA/*")

    corpus = []
    for file in file_list:
        print(file)
        with open(file, encoding="utf-8") as f:
            raw = f.readlines()
            for r in raw:
                if r.find("</doc>") == 0:
                    corpus.append("")
                clean = text_cleaning(r)  # text cleaning
                if clean != "":
                    sentences = jp_sentence.tokenize(clean)  # split sentence
                    if sentences:
                        for s in sentences:
                            corpus.append(s)

    with open("./jawiki/jawiki_corpus.txt", "w", encoding="utf-8") as f:
        for s in corpus:
            f.write("%s\n" % s)


def bert_sentencepiece():
    spm.SentencePieceTrainer.train(input="./jawiki/jawiki_corpus.txt", model_prefix="jawiki", vocab_size=num_word,
                                   bos_id=-1, eos_id=-1, user_defined_symbols="[CLS],[SEP],[MASK]")


def masked_language_model(word_list):
    num_replace = max(1, int(len(word_list) * masked_rate))

    masked_idx_list = random.sample(range(len(word_list)), num_replace)

    word_list_masked = []
    for idx in range(len(word_list)):
        if idx in masked_idx_list:
            if rng.random() < 0.8:  # 80% [MASK]
                word_list_masked.append(MASK)
            else:
                if rng.random() < 0.5:  # 10% random
                    word_list_masked.append(random.randint(4, num_word - 1))
                else:  # 10% unchanged
                    word_list_masked.append(word_list[idx])
        else:
            word_list_masked.append(word_list[idx])

    return word_list, word_list_masked, masked_idx_list


def sentence_order_prediction(sent1, sent2):
    word_list1, word_list1_masked, masked_idx_list1 = masked_language_model(sent1)
    word_list2, word_list2_masked, masked_idx_list2 = masked_language_model(sent2)

    # first sentence
    word_list1.insert(0, CLS)
    word_list1.append(SEP)
    word_list1_masked.insert(0, CLS)
    word_list1_masked.append(SEP)

    # next sentence
    word_list2.append(SEP)
    word_list2_masked.append(SEP)

    masked = word_list1_masked + word_list2_masked
    answer = word_list1 + word_list2
    segment = [0] * len(word_list1) + [1] * len(word_list2)

    index = list(np.array(masked_idx_list1) + 1) + list(np.array(masked_idx_list2) + len(word_list1))
    masked_idx = np.zeros(len(masked))
    masked_idx[index] = 1

    return masked, answer, segment, masked_idx


def bert_pretraining():
    #
    # sentence piece
    #
    spm_model = SentencePiece(spm_path)

    with open("./jawiki/jawiki_corpus.txt", encoding="utf-8") as f:
        corpus = f.readlines()

    #
    # masked, answer, segment, index, and label
    #
    num_samples = 0
    with open("./jawiki/train_bert_corpus.txt", "w") as map_file:
        for c in range(len(corpus) - 1):
            sent1 = corpus[c]
            sent2 = corpus[c + 1]
            if sent1 == "\n" or sent2 == "\n":
                continue
            else:
                if (len(spm_model.encode(sent1) + spm_model.encode(sent2)) + 2) > max_seq_len:
                    continue
                else:
                    # forward
                    masked, answer, segment, masked_idx =\
                        sentence_order_prediction(spm_model.encode(sent1), spm_model.encode(sent2))

                    map_file.write("{} |masked {}:1\t|answer {}:1\t|segment {}:1\t|index {}\t|label {}\n".format(
                        num_samples, masked[0], answer[0], segment[0], masked_idx[0], 1))  # first row

                    for msk, ans, seg, idx in zip(masked[1:], answer[1:], segment[1:], masked_idx[1:]):
                        map_file.write("{} |masked {}:1\t|answer {}:1\t|segment {}:1\t|index {}\n".format(
                            num_samples, msk, ans, seg, idx))

                    num_samples += 1

                    # swap order
                    masked, answer, segment, masked_idx =\
                        sentence_order_prediction(spm_model.encode(sent2), spm_model.encode(sent1))

                    map_file.write("{} |masked {}:1\t|answer {}:1\t|segment {}:1\t|index {}\t|label {}\n".format(
                        num_samples, masked[0], answer[0], segment[0], masked_idx[0], 0))  # first row

                    for msk, ans, seg, idx in zip(masked[1:], answer[1:], segment[1:], masked_idx[1:]):
                        map_file.write("{} |masked {}:1\t|answer {}:1\t|segment {}:1\t|index {}\n".format(
                            num_samples, msk, ans, seg, idx))

                    num_samples += 1

    print("\nNumber of samples:", num_samples)


if __name__ == "__main__":
    bert_preprocessing()

    bert_sentencepiece()
    
