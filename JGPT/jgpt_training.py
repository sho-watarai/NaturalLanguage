import cntk as C
import cntkx as Cx
import os
import pandas as pd
import sentencepiece as spm

from cntk.layers import Dense, Embedding, LayerNormalization
from cntk.layers.blocks import _inject_name
from cntkx.learners import CyclicalLearningRate

num_head = 12
num_hidden = 1024
num_stack = 12
num_word = 32000

iteration = 300000
max_seq_len = 128
minibatch_size = 256
num_samples = 23704

sample_size = 12
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


def ScaledDotProductAttention(obey_sequence_order=None, max_seq_len=None, name=''):
    """ Scaled Dot Product Attention """
    @C.Function
    def attention(query, key, value):
        dk = C.sqrt(C.reduce_sum(C.ones_like(query)))  # dk: [#, *] [1, ] and value = int(dim_of_query)

        unpacked_key = C.sequence.unpack(key, padding_value=0, no_mask_output=True)  # [#] [-3, key_dim]
        unpacked_value = C.sequence.unpack(value, padding_value=0, no_mask_output=True)  # [#] [-3, value_dim]

        broadcasted_key = C.sequence.broadcast_as(unpacked_key, query)  # [#, *] [-3, key_dim]
        scaled = C.times_transpose(query, broadcasted_key) / dk

        if obey_sequence_order and max_seq_len:
            unpacked_scaled, scaled_mask = C.sequence.unpack(scaled, padding_value=0).outputs

            minus_inf = C.constant(-1e+30)
            valid_connections = C.Constant(np.tril(np.ones((max_seq_len, max_seq_len)), k=0))  # [] [max_seq, max_seq]
            valid_connections = C.reconcile_dynamic_axes(valid_connections, unpacked_scaled)  # [#] [max_seq, max_seq]
            valid_connections = C.crop_manual(valid_connections, unpacked_scaled, 0, 0)  # [#] [-3, -3]
            unpacked_scaled = C.element_select(valid_connections, unpacked_scaled, minus_inf)  # [#] [-3, -3]
            scaled = C.to_sequence_like(unpacked_scaled, query)  # [#, *] [-3]

        elif obey_sequence_order and not max_seq_len:
            raise ValueError("max_seq_len must be defined when obey_sequence_order is True")

        attention_weights = C.softmax(scaled, axis=-1)
        attention_weights = C.layers.Label('attention_weights')(attention_weights)

        attended = C.times(attention_weights, C.sequence.broadcast_as(unpacked_value, query))  # [#, *] [value_dim,]
        return attended

    return _inject_name(attention, name)


def AlbertMultiHeadAttention(num_heads, model_dim, key_linear, query_linear, value_linear, concat_linear,
                             obey_sequence_order=None, max_seq_len=None, name=''):
    """ Multi Head Attention Cross-layer parameter sharing """
    head_dim = model_dim // num_heads

    scaled_dot_product_attention = [ScaledDotProductAttention(
        obey_sequence_order, max_seq_len, name=name + "_%d" % (i + 1)) for i in range(num_heads)]

    @C.Function
    def inner(query, key, value):
        queries = [query_linear[i](C.slice(query, 0, i * head_dim, (i + 1) * head_dim)) for i in range(num_heads)]
        keys = [key_linear[i](C.slice(key, 0, i * head_dim, (i + 1) * head_dim)) for i in range(num_heads)]
        values = [value_linear[i](C.slice(value, 0, i * head_dim, (i + 1) * head_dim)) for i in range(num_heads)]

        attention_outputs = [
            scaled_dot_product_attention[i](q, k, v) for i, (q, k, v) in enumerate(zip(queries, keys, values))]

        return concat_linear(C.splice(*attention_outputs))

    return _inject_name(inner, name)


def BertPositionwiseFeedForward(outer_dim, inner_dim, dropout_rate, name=''):
    inner_dense = Dense(inner_dim, init=C.normal(0.02), name='inner')
    outer_dense = Dense(outer_dim, init=C.normal(0.02), name='outer')
    dropout = Dropout(dropout_rate)

    @C.BlockFunction('BertPositionwiseFeedForward', name)
    def bert_positionwise_feedforward(x):
        return outer_dense(dropout(Cx.gelu_fast(inner_dense(x))))

    return bert_positionwise_feedforward


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
    
