import cntk as C
import numpy as np
import sentencepiece as spm

num_word = 32000

BOS = 1
EOS = 2
UNK = 0

MAX = 70

spm_path = "./twitter.model"


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
    # Sentence Piece
    #
    spm_model = SentencePiece(spm_path)

    model = C.load_model("./stsa.model")

    #
    # ChatBot Application
    #
    while True:
        query = input(">")
        if query == "quit":
            break

        query = spm_model.encode(query)
        query.insert(0, BOS)
        query.append(EOS)

        query = np.identity(num_word, dtype="float32")[query]
        reply = np.identity(num_word, dtype="float32")[BOS].reshape(1, -1)
        dummy = np.identity(num_word, dtype="float32")[EOS].reshape(1, -1)

        for _ in range(MAX):
            prob = model.eval({model.arguments[0]: query, model.arguments[1]: np.vstack((reply, dummy))})[0]
            pred = np.identity(num_word, dtype="float32")[prob.argmax(axis=1)[-1]].reshape(1, -1)
            reply = np.concatenate((reply, pred), axis=0)
            if prob.argmax(axis=1)[-1] in [EOS, 5, 6, 8]:  # ! 。w
                break

        response = spm_model.decode([int(i) for i in reply.argmax(axis=1)])

        print(">>", response)
        
