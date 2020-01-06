import cntk as C
import matplotlib.pyplot as plt
import numpy as np
import pickle

from sklearn.manifold import TSNE

fp.rcParams["font.family"] = "Yu Gothic"

with open("./word2id.pkl", "rb") as f:
    word2id = pickle.load(f)

with open("./id2word.pkl", "rb") as f:
    id2word = pickle.load(f)


def word_embedding(embedding):
    plt.figure()
    for j, ppl in enumerate([5, 10, 20, 30, 50]):
        tsne = TSNE(n_components=2, perplexity=ppl, init="pca", n_iter=1000)
        vec = tsne.fit_transform(embedding)

        plt.subplot(1, 5, j + 1)
        plt.scatter(vec[:, 0], vec[:, 1], c="b", edgecolors="b")
        plt.title("perplexity=%d" % ppl)
    plt.show()


def word_similarity(query, word2id, id2word, embedding, topN=5):
    print("[similarity] " + query)
    query_id = word2id[query]
    query_vec = np.repeat(embedding[query_id].reshape(1, -1), len(word2id), axis=0)
    num_hidden = embedding.shape[1]

    x = C.input_variable(shape=(num_hidden,))
    y = C.input_variable(shape=(num_hidden,))

    similarity = C.cosine_distance(x, y).eval({x: embedding, y: query_vec})

    topN_id = similarity.argsort()[::-1][1:(topN + 1)]
    for i in topN_id:
        print("%s\t:%.2f" % (id2word[i], similarity[i]))


def word_analogy(a, b, c, word2id, id2word, embedding, topN=5, answer=None):
    for word in (a, b, c):
        if word not in word2id:
            print("%s is not found" % word)
            return

    print("\n[analogy] " + a + " - " + b + " + " + c + " = ?")
    a_vec, b_vec, c_vec = embedding[word2id[a]], embedding[word2id[b]], embedding[word2id[c]]
    query_vec = b_vec - a_vec + c_vec
    query_vec /= np.sqrt(query_vec * query_vec).sum()

    similarity = np.dot(embedding, query_vec)

    if answer is not None:
        print("==>" + answer + ":" + str(np.dot(embedding[word2id[answer]], query_vec)))

    topN_id = similarity.argsort()[::-1][1:(topN + 4)]
    for j, i in enumerate(topN_id):
        if id2word[i] in {a, b, c}:
            continue
        else:
            print("%s\t:%.2f" % (id2word[i], similarity[i]))

        if j >= 5:
            break


if __name__ == "__main__":
    model = C.load_model("./skipgram.model")

    E = model.parameters[0].value

    word_similarity("魔法", word2id, id2word, E)

    word_analogy("葉月", "蓮", "仁", word2id, id2word, E)

    word_embedding(E)
    
