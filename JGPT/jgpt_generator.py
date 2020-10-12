import cntk as C
import figpy as fp
import numpy as np
import sentencepiece as spm

num_word = 32000

fp.rcParams["font.family"] = "Yu Gothic"
fp.rcParams["font.size"] = 15

BOS = 1
EOS = 2
MAX = 128


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


if __name__ == "__main__":
    #
    # sentence piece and model
    #
    spm_model = SentencePiece("./corpus.model")
    gpt_model = C.load_model("./jgpt.model")

    #
    # text generation
    #
    ja = input(">")
    ja = spm_model.encode(ja)
    ja.insert(0, BOS)

    ja = np.identity(num_word, dtype="float32")[ja]
    dummy = np.identity(num_word, dtype="float32")[EOS].reshape(1, -1)

    for _ in range(MAX - len(ja) - 1):
        prob = gpt_model.eval({gpt_model.arguments[0]: np.vstack((ja, dummy))})[0]
        pred = np.identity(num_word, dtype="float32")[prob.argmax(axis=1)[-1]].reshape(1, -1)
        ja = np.concatenate((ja, pred), axis=0)
        if prob.argmax(axis=1)[-1] == EOS:
            break

    text = spm_model.decode([int(i) for i in ja.argmax(axis=1)])
    print(text)
    
