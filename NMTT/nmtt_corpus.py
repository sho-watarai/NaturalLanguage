import re
import sentencepiece as spm

from itertools import zip_longest

UNK = 0
BOS = 1
EOS = 2

data_file = "train"


class SentencePiece:
    def __init__(self, spm_path):
        self.model = spm.SentencePieceProcessor()
        self.model.Load(spm_path)

    def encode(self, string):
        return self.model.EncodeAsIds(string)

    def decode(self, ids):
        return self.model.DecodeIds(ids)


def japanese_cleaning(s):
    s = re.sub(r"・+・", "・", s)
    s = re.sub(r"[.]+", "…", s)
    s = re.sub(r"[^a-z0-9ぁ-んァ-ン一-龥、。!?「」ー…＆・]", "", s)

    return s


def english_cleaning(s):
    s = re.sub(r"[\[+\]]", "", s)
    s = re.sub(r"[.+.]+", ".", s)
    s = re.sub(r";", " ", s)
    s = re.sub(r"[ ]+", " ", s)

    return s


def jesc_preprocessing(date_file):
    with open("./JESC/{}".format(date_file), encoding="utf-8") as f:
        train = f.readlines()

    english, japanese = [], []

    for text in train:
        en, ja = text[:-1].split("\t")

        en_clean = english_cleaning(en)
        ja_clean = japanese_cleaning(ja)

        if not re.findall(r"[ぁ-んァ-ン]", ja_clean):  # not japanese maybe chinese
            print(ja_clean)
            continue

        english.append(en_clean)
        japanese.append(ja_clean)

    with open("./JESC/{}.english.txt".format(date_file), "w", encoding="utf-8") as f:
        for en in english:
            f.write("%s\n" % en)

    with open("./JESC/{}.japanese.txt".format(date_file), "w", encoding="utf-8") as f:
        for ja in japanese:
            f.write("%s\n" % ja)


def jesc_sentencepeice(data_file):
    #
    # sentence piece
    #
    spm_model_en = SentencePiece("./english.model")
    spm_model_ja = SentencePiece("./japanese.model")

    with open("./JESC/{}.english.txt".format(data_file), encoding="utf-8") as f:
        english = f.readlines()

    with open("./JESC/{}.japanese.txt".format(data_file), encoding="utf-8") as f:
        japanese = f.readlines()

    assert len(english) == len(japanese)

    MAX = 0
    japanese_english = []
    for i in range(len(japanese)):
        ja = spm_model_ja.encode(japanese[i])

        en = spm_model_en.encode(english[i])
        en.insert(0, BOS)
        en.append(EOS)

        max_seq_len = max(len(ja), len(en))
        if max_seq_len > MAX:
            MAX = max_seq_len

        japanese_english.append([ja, en])

    #
    # japanese and english
    #
    num_samples = 0
    with open("./{}_jesc_corpus.txt".format(data_file), "w") as ctf_file:
        for i, (japanese, english) in enumerate(japanese_english):
            for (ja, en) in zip_longest(japanese, english, fillvalue=""):
                if ja == "":
                    ctf_file.write("{} |english {}:1\n".format(i, en))
                elif en == "":
                    ctf_file.write("{} |japanese {}:1\n".format(i, ja))
                else:
                    ctf_file.write("{} |japanese {}:1\t|english {}:1\n".format(i, ja, en))

            num_samples += 1
            if num_samples % 10000 == 0:
                print("Now %d samples..." % num_samples)

    print("\nNumber of samples", num_samples)
    print("\nMaximum Sequence Length", MAX)


if __name__ == "__main__":
    jesc_preprocessing(data_file)

    jesc_sentencepeice(data_file)
    

