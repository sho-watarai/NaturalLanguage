
import emoji
import glob
import re
import sentencepiece as spm

from itertools import zip_longest

BOS = 1
EOS = 2
UNK = 0

dir_file = "./twitter"
spm_path = "./twitter.model"


class SentencePiece:
    def __init__(self, spm_path):
        self.model = spm.SentencePieceProcessor()
        self.model.Load(spm_path)

    def encode(self, string):
        return self.model.EncodeAsIds(string)

    def decode(self, ids):
        return self.model.DecodeIds(ids)


def text_cleaning(s):
    #
    # twitter specific
    #
    s = re.sub(r"http\S+", "", s)  # remove https
    s = re.sub(r"\@[a-z0-9-_][a-z0-9-_]* ", "", s)  # remove @tweet
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
    s = re.sub(r"w+w", "w", s)
    s = re.sub(r" + ", "", s)

    #
    # remove emoji and kaomoji
    #
    s = "".join(["" if c in emoji.UNICODE_EMOJI else c for c in s])  # remove emoji
    s = re.sub(r"（.*）.*", "", s)  # remove kaomoji

    s = re.sub(r"[^ a-z0-9ぁ-んァ-ン一-龥、。！？ー…＆]", "", s)
    s = re.sub(r"\A[^a-z0-9ぁ-んァ-ン一-龥]*", "", s)

    return s


def stsa_preprocessing():
    corpus = []

    file_list = glob.glob(dir_file + "/*.txt")
    for file in file_list:
        with open(file, encoding="UTF-8") as f:
            corpus += f.readlines()
    print("Number of tweet and reply:", len(corpus) // 2)

    with open("./twitter.txt", "w", encoding="utf-8") as f:
        for idx in range(len(corpus) // 2):
            tweet = text_cleaning(corpus[2 * idx])
            reply = text_cleaning(corpus[2 * idx + 1])

            if len(tweet) == 0 or len(reply) == 0:
                continue
            else:
                f.write("%s\n" % tweet)
                f.write("%s\n" % reply)


def stsa_sentencepiece():
    #
    # Sentence Piece
    #
    spm_model = SentencePiece(spm_path)

    with open("./twitter.txt", encoding="utf-8") as f:
        corpus = f.readlines()

    tweet_reply = []
    for idx in range(len(corpus) // 2):
        tweet = spm_model.encode(corpus[2 * idx][:-1])
        tweet.insert(0, BOS)
        tweet.append(EOS)

        reply = spm_model.encode(corpus[2 * idx + 1][:-1])
        reply.insert(0, BOS)
        reply.append(EOS)

        tweet_reply.append([tweet, reply])

    #
    # tweet and reply
    #
    num_samples = 0
    with open("./tweet_reply_corpus.txt", "w") as ctf_file:
        for i, (tweet, reply) in enumerate(tweet_reply):
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


if __name__ == "__main__":
    stsa_preprocessing()

    stsa_sentencepiece()
    
