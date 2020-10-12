import cntk as C
import cntkx as Cx
import os
import pandas as pd
import sentencepiece as spm

from cntk.layers import Dense, Embedding, LayerNormalization
from cntkx.learners import CyclicalLearningRate

num_head = 12
num_hidden = 1024
num_stack = 12
num_word = 32000

iteration = 1000000
max_seq_len = 128
minibatch_size = 256
num_samples = 20216

sample_size = 16
step_size = num_samples // sample_size * 10


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


def create_reader(path, is_train):
    return C.io.MinibatchSource(C.io.CTFDeserializer(path, C.io.StreamDefs(
        token=C.io.StreamDef(field="token", shape=num_word, is_sparse=True))),
                                randomize=is_train, max_sweeps=C.io.INFINITELY_REPEAT if is_train else 1)


def jgpt_base(h):
    token_embed_factor = Embedding(num_hidden // 8, init=C.normal(0.02))  # V x E
    token_embed_hidden = Embedding(num_hidden, init=C.normal(0.02))  # E x H

    position_embed_factor = Embedding(num_hidden // 8, init=C.normal(0.01))  # V x E
    position_embed_hidden = Embedding(num_hidden, init=C.normal(0.01))  # E x H

    h = token_embed_hidden(token_embed_factor(h)) +\
        position_embed_hidden(position_embed_factor(h + Cx.sequence.position(h)))

    # cross-layer parameter sharing
    query_linear = [Dense(num_hidden // num_head, init=C.normal(0.02)) for _ in range(num_head)]
    key_linear = [Dense(num_hidden // num_head, init=C.normal(0.02)) for _ in range(num_head)]
    value_linear = [Dense(num_hidden // num_head, init=C.normal(0.02)) for _ in range(num_head)]
    concat_linear = Dense(num_hidden, init=C.normal(0.02))

    feedforward = BertPositionwiseFeedForward(num_hidden, num_hidden * 4, dropout_rate=0.1)

    #
    # transformer decoder
    #
    for i in range(num_stack):
        norm = LayerNormalization()(h)
        mha = AlbertMultiHeadAttention(num_head, num_hidden, query_linear, key_linear, value_linear, concat_linear,
                                       obey_sequence_order=True, max_seq_len=max_seq_len,
                                       name="dec%d" % (i + 1))(norm, norm, norm)  # masked self attention
        h = mha + h

        norm = LayerNormalization()(h)
        ffn = feedforward(norm)

        h = ffn + h

    h = LayerNormalization()(h)
    h = Dense(num_word, activation=None, init=C.normal(0.02))(h)

    return h


def write_text(model, data, token, spm_model):
    sources = token * 1
    queries = sources.eval({sources.arguments[0]: data[token].data})
    outputs = model.eval({token: data[token].data})

    for query, output in zip(queries, outputs):
        que = spm_model.decode([int(q) for q in query.argmax(axis=1)])
        print(que, end=" -> ")

        out = spm_model.decode([int(o) for o in output.argmax(axis=1)])
        print(out)

    return que, out


if __name__ == "__main__":
    #
    # sentence piece
    #
    spm_model = SentencePiece("./corpus.model")

    #
    # built-in reader
    #
    train_reader = create_reader("./train_jgpt_corpus.txt", is_train=True)

    #
    # token, label, and model
    #
    token = C.sequence.input_variable(shape=(num_word,), sequence_axis=C.Axis("Token"))

    model = jgpt_base(C.sequence.slice(token, 0, -1))

    input_map = {token: train_reader.streams.token}

    #
    # loss function and perplexity
    #
    loss = C.cross_entropy_with_softmax(model, C.sequence.slice(token, 1, 0))

    #
    # optimizer and cyclical learning rate
    #
    learner = C.adam(model.parameters, lr=C.learning_parameter_schedule_per_sample(0.1), momentum=0.9,
                     gradient_clipping_threshold_per_sample=sample_size, gradient_clipping_with_truncation=True)
    clr = CyclicalLearningRate(learner, base_lr=1e-8, max_lr=1e-4,
                               ramp_up_step_size=step_size, minibatch_size=sample_size)
    progress_printer = C.logging.ProgressPrinter(tag="Training")

    trainer = C.Trainer(model, (loss, None), [learner], [progress_printer])

    C.logging.log_number_of_parameters(model)

    #
    # training
    #
    logging = {"step": [], "loss": [], "input": [], "model": []}
    for step in range(iteration):
        data = train_reader.next_minibatch(minibatch_size, input_map=input_map)

        trainer.train_minibatch(data)

        clr.batch_step()

        if step % (iteration // 100) == 0:
            step_loss = trainer.previous_minibatch_loss_average
            
            que, out = write_text(model, data, token, spm_model)

            #
            # loss logging
            #
            logging["step"].append(step)
            logging["loss"].append(step_loss)
            logging["input"].append(que)
            logging["model"].append(out)

        trainer.summarize_training_progress()

    #
    # save model and logging
    #
    model.save("./jgpt.model")
    print("Saved model.")

    df = pd.DataFrame(logging)
    df.to_csv("./jgpt.csv", index=False)
    print("Saved logging.")
    
