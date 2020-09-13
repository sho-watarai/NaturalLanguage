import glob
import MeCab
import nltk
import pickle
import re
import sentencepiece as spm

jp_sentence = nltk.RegexpTokenizer("[^。]*[。]")

num_word = 32000

CLS = 1  # [CLS]
SEP = 2  # [SEP]
MASK = 3  # [MASK]

spm_path = "./jawiki.model"

max_seq_len = 128

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
    s = re.sub(r"http\S+", "", s)  # remove html
    s = re.sub("[0-9].*", "", s)  # remove time

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


def bert_finetuning():
    #
    # MeCab with NEologd and sentence piece
    #
    mecab = MeCabTokenizer()
    spm_model = SentencePiece(spm_path)

    doc_train = []
    doc_val = []

    dir_list = glob.glob("../Doc2Vec/text/*")
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

                document = ""
                for t in text:
                    text_clean = text_cleaning(t)
                    word_list = mecab.tokenize(text_clean)
                    for w in word_list:
                        if w in stop_words:
                            word_list.remove(w)  # remove stop words

                    for word in word_list:
                        document += word

                tokenized_list = spm_model.encode(document)[:max_seq_len - 2]
                tokenized_list.insert(0, CLS)
                tokenized_list.append(SEP)

                tokenized_list.append(label)

            doc_train.append(tokenized_list)

        #
        # validation data
        #
        for file in file_list[-10:]:
            with open(file, encoding="utf-8") as f:
                text = f.readlines()

                document = ""
                for t in text:
                    text_clean = text_cleaning(t)
                    word_list = mecab.tokenize(text_clean)
                    for w in word_list:
                        if w in stop_words:
                            word_list.remove(w)  # remove stop words

                    for word in word_list:
                        document += word

                tokenized_list = spm_model.encode(document)[:max_seq_len - 2]
                tokenized_list.insert(0, CLS)
                tokenized_list.append(SEP)

                tokenized_list.append(label)

            doc_val.append(tokenized_list)

    #
    # document and label
    #
    num_samples = 0
    with open("./train_doc2vec_map.txt", "w") as map_file:
        for i, doc in enumerate(doc_train):
            map_file.write("{} |word {}:1\t|segment {}:1\t|label {}:1\n".format(i, doc[0], 0, doc[-1]))
            for j in range(1, len(doc) - 1):
                map_file.write("{} |word {}:1\t|segment {}:1\n".format(i, doc[j], 0))

            num_samples += 1

    print("\nNumber of training samples", num_samples)

    num_samples = 0
    with open("./val_doc2vec_map.txt", "w") as map_file:
        for i, doc in enumerate(doc_val):
            map_file.write("{} |word {}:1\t|segment {}:1\t|label {}:1\n".format(i, doc[0], 0, doc[-1]))
            for j in range(1, len(doc) - 1):
                map_file.write("{} |word {}:1\t|segment {}:1\n".format(i, doc[j], 0))

            num_samples += 1

    print("Number of validation samples", num_samples)


if __name__ == "__main__":
    bert_finetuning()
    
