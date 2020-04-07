import emoji
import glob
import MeCab
import pickle
import re

from collections import Counter
from itertools import zip_longest

dir_file = "./twitter"

BOS = "<BOS>"
EOS = "<EOS>"
UNK = "<UNK>"

neologd_path = "C:/Users/usr/AppData/Local/Packages/KaliLinux.54290C8133FEE_ey8k8hqnwqnmg/LocalState/rootfs" \
               "/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd"


class MeCabTokenizer:
    def __init__(self):
        self.tokenizer = MeCab.Tagger("-O wakati -d %s" % neologd_path)

    def tokenize(self, string):
        return self.tokenizer.parse(string).split(" ")[:-1]


def text_cleaning(s, verbose=False):
    if verbose:
        print("Before", s)

    #
    # twitter specific
    #
    s = re.sub(r"http\S+", "", s)  # remove https
    s = re.sub(r"\@[a-z0-9-_][a-z0-9-_]*", "", s)  # remove @tweet
    s = re.sub(r"\#.+", "", s)  # remove #tag

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

    #
    # remove emoji and kaomoji
    #
    s = "".join(["" if c in emoji.UNICODE_EMOJI else c for c in s])  # remove emoji
    s = re.sub(r"（.*）.*", "", s)  # remove kaomoji

    s = re.sub(r"[^a-z0-9ぁ-んァ-ン一-龥、。！？ー…＆／]", "", s)

    if verbose:
        print("After", s)

    return s


def create_word2id(captions):
    counter = Counter()
    for c in captions:
        counter.update(c)

    print("Number of total words:", len(counter))
    word_counts = [x for x in counter.items() if x[1] >= 10]  # less 10 count word is not common word
    word_counts.sort(key=lambda x: x[1], reverse=True)
    print("Number of words:", len(word_counts) + 1)  # plus 1 is <UNK>
    word_list = [x[0] for x in word_counts]
    word_list.remove(BOS)
    word_list.remove(EOS)
    word_list.insert(0, BOS)
    word_list.insert(0, EOS)

    word_list.append(UNK)
    word2id = dict([(x, y) for (y, x) in enumerate(word_list)])
    id2word = dict([(x, y) for (x, y) in enumerate(word_list)])

    return word2id, id2word


if __name__ == "__main__":
    mecab = MeCabTokenizer()

    corous = []
    file_list = glob.glob(dir_file + "/*.txt")
    for file in file_list:
        with open(file, encoding="UTF-8") as f:
            corous += f.readlines()
    print("Number of tweet and reply:", len(corous) // 2)
    print()

    tweet_list = []
    for t in corous:
        t = text_cleaning(t)
        
        word_list = mecab.tokenize(t)
        word_list.insert(0, BOS)
        word_list.append(EOS)

        tweet_list.append(word_list)

    #
    # word2id and id2word dictionary
    #
    word2id, id2word = create_word2id(tweet_list)

    with open("./word2id.pkl", "wb") as f:
        pickle.dump(word2id, f)
    print("\nSaved word2id.pkl")

    with open("./id2word.pkl", "wb") as f:
        pickle.dump(id2word, f)
    print("Saved id2word.pkl\n")

    #
    # create tweet and reply corpus
    #
    tweet_reply = []
    for idx in range(len(tweet_list) // 2):
        tweet1 = tweet_list[2 * idx]
        tweet2 = tweet_list[2 * idx + 1]

        if 2 < len(tweet1) and 2 < len(tweet2):
            tweet = [word2id[w] if w in word2id else word2id[UNK] for w in tweet1]
            reply = [word2id[w] if w in word2id else word2id[UNK] for w in tweet2]
            
            tweet_reply.append([tweet, reply[:-1], reply[1:]])

    #
    # tweet and reply
    #
    num_samples = 0
    with open("./tweet_reply_corpus.txt", "w") as ctf_file:
        for i, (tweet, reply, target) in enumerate(tweet_reply):
            for (twt, rep) in zip_longest(tweet, reply, fillvalue=""):
                if twt == "":
                    ctf_file.write("{} |reply {}:1\n".format(i, rep))
                elif rep == "":
                    ctf_file.write("{} |tweet {}:1\n".format(i, twt))
                else:
                    ctf_file.write("{} |tweet {}:1\t|reply {}:1\n".format(i, twt, rep))

            num_samples += 1
            if num_samples % 10000 == 0:
                print("Now %d samples..." % num_samples)

    print("\nNumber of samples", num_samples)
    
