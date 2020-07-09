import cntk as C
import numpy as np
import sentencepiece as spm

num_word = 32000

UNK = 0
BOS = 1
EOS = 2

MAX = 97


class SentencePiece:
    def __init__(self, spm_path):
        self.model = spm.SentencePieceProcessor()
        self.model.Load(spm_path)

    def encode(self, string):
        return self.model.EncodeAsIds(string)

    def decode(self, ids):
        return self.model.DecodeIds(ids)


if __name__ == "__main__":
    #
    # sentence piece
    #
    en_model = SentencePiece("./english.model")
    ja_model = SentencePiece("./japanese.model")

    model = C.load_model("./nmtt.model")

    #
    # machine translation Application
    #
    while True:
        ja = input(">")
        if ja == "quit":
            break
        ja = ja_model.encode(ja)

        ja = np.identity(num_word, dtype="float32")[ja]
        en = np.identity(num_word, dtype="float32")[BOS].reshape(1, -1)
        dummy = np.identity(num_word, dtype="float32")[EOS].reshape(1, -1)

        for _ in range(MAX):
            prob = model.eval({model.arguments[1]: ja, model.arguments[0]: np.vstack((en, dummy))})[0]
            pred = np.identity(num_word, dtype="float32")[prob.argmax(axis=1)[-1]].reshape(1, -1)
            en = np.concatenate((en, pred), axis=0)
            if prob.argmax(axis=1)[-1] == EOS:
                break

        response = en_model.decode([int(i) for i in en.argmax(axis=1)])

        print(">>", response)
        
