import cntk as C
import cntkx as Cx
import os
import pandas as pd

from cntk.layers import Dense, Dropout, LayerNormalization
from cntk.layers.blocks import _inject_name
from cntkx.layers import Embedding, PositionalEmbedding
from cntkx.learners import CyclicalLearningRate

num_head = 12
num_hidden = 768
num_segment = 2
num_stack = 12
num_word = 32000

iteration = 3000000
max_seq_len = 128
minibatch_size = 512
num_samples = 36862478

sample_size = 12
step_size = num_samples // sample_size * 10


def create_reader(path, is_train):
    return C.io.MinibatchSource(C.io.CTFDeserializer(path, C.io.StreamDefs(
        masked=C.io.StreamDef(field="masked", shape=num_word, is_sparse=True),
        answer=C.io.StreamDef(field="answer", shape=num_word, is_sparse=True),
        segment=C.io.StreamDef(field="segment", shape=num_segment, is_sparse=True),
        index=C.io.StreamDef(field="index", shape=1, is_sparse=False),
        label=C.io.StreamDef(field="label", shape=1, is_sparse=False))),
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


def AlbertEmbedding(hidden_dim, max_seq_length, name=''):
    token_embed_factor = Embedding(hidden_dim // 6, init=C.normal(0.02))  # V x E
    token_embed_hidden = Embedding(hidden_dim, init=C.normal(0.02))  # E x H

    segment_embed_factor = Embedding(hidden_dim // 6, init=C.normal(0.02))  # V x E
    segment_embed_hidden = Embedding(hidden_dim, init=C.normal(0.02))  # E x H

    position_embed = PositionalEmbedding(hidden_dim, max_seq_length, init=C.normal(0.02))

    @C.BlockFunction('AlbertEmbedding', name)
    def bert_embedding(word, segment):
        h = token_embed_hidden(token_embed_factor(word)) + segment_embed_hidden(segment_embed_factor(segment))
        h = h + position_embed(word)
        return h

    return bert_embedding


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
        return dropout(outer_dense(Cx.gelu_fast(inner_dense(x))))

    return bert_positionwise_feedforward


def albert_base(token, segment):
    h = AlbertEmbedding(num_hidden, max_seq_len)(token, segment)

    # cross-layer parameter sharing
    query_linear = [Dense(num_hidden // num_head, init=C.normal(0.02)) for _ in range(num_head)]
    key_linear = [Dense(num_hidden // num_head, init=C.normal(0.02)) for _ in range(num_head)]
    value_linear = [Dense(num_hidden // num_head, init=C.normal(0.02)) for _ in range(num_head)]
    concat_linear = Dense(num_hidden, init=C.normal(0.02))

    feedforward = BertPositionwiseFeedForward(num_hidden, num_hidden * 4, dropout_rate=0.1)

    #
    # transformer encoder
    #
    for i in range(num_stack):
        norm = LayerNormalization()(h)
        mha = AlbertMultiHeadAttention(num_head, num_hidden, query_linear, key_linear, value_linear, concat_linear,
                                       name="enc%d" % (i + 1))(norm, norm, norm)  # self attention
        mha = Dropout(0.1)(mha)

        h = mha + h

        norm = LayerNormalization()(h)
        ffn = feedforward(norm)

        h = ffn + h

    h_encode = _inject_name(h, "encode")
    h_pooler = Dense(num_hidden, activation=C.tanh, init=C.normal(0.02), name="pooler")(h)

    return h_encode, h_pooler


def bert_masked_lm(encode):
    """ Masked Language Model """
    h = Dense(num_hidden, activation=Cx.gelu_fast, init=C.normal(0.02))(encode)
    h = LayerNormalization()(h)
    return Dense(num_word, activation=None, init=C.normal(0.02))(h)


def bert_sentence_prediction(pooler):
    """ Sentence Prediction """
    h = C.sequence.unpack(pooler, padding_value=0, no_mask_output=True)[0, :]
    return Dense(1, activation=C.sigmoid, init=C.normal(0.02))(h)


if __name__ == "__main__":
    #
    # built-in reader
    #
    train_reader = create_reader("./train_bert_corpus.txt", is_train=True)

    #
    # token, segment, answer, index, and label
    #
    token = C.sequence.input_variable(num_word)
    segment = C.sequence.input_variable(num_segment)

    answer = C.sequence.input_variable(num_word)
    index = C.sequence.input_variable(1)  # masked language model
    label = C.input_variable(1)  # sentence prediction

    encode, pooler = albert_base(token, segment)

    #
    # unsupervised pre-training
    #
    mask_lm = bert_masked_lm(encode)
    sentence_prediction = bert_sentence_prediction(pooler)

    #
    # loss function
    #
    masked_lm_loss = C.sequence.reduce_sum(index * C.cross_entropy_with_softmax(mask_lm, answer)) / C.sequence.reduce_sum(index)
    sentence_prediction_loss = C.binary_cross_entropy(sentence_prediction, label)

    loss = masked_lm_loss + sentence_prediction_loss

    C.logging.log_number_of_parameters(loss)

    #
    # optimizer and cyclical learning rate
    #
    learner = C.adam(loss.parameters, lr=C.learning_parameter_schedule_per_sample(0.1), momentum=0.9,
                     gradient_clipping_threshold_per_sample=minibatch_size, gradient_clipping_with_truncation=True)
    clr = CyclicalLearningRate(learner, base_lr=1e-8, max_lr=1e-4,
                               ramp_up_step_size=step_size, minibatch_size=minibatch_size)
    progress_printer = C.logging.ProgressPrinter(tag="Training")

    trainer = C.Trainer(loss, (loss, None), [learner], [progress_printer])

    input_map = {token: train_reader.streams.masked, segment: train_reader.streams.segment,
                 answer: train_reader.streams.answer, index: train_reader.streams.index,
                 label: train_reader.streams.label}
    
    #
    # training
    #
    logging = {"step": [], "loss": []}
    for step in range(iteration):
        data = train_reader.next_minibatch(minibatch_size, input_map=input_map)

        trainer.train_minibatch(data)

        clr.batch_step()

        if step % (iteration // 100) == 0:
            step_loss = trainer.previous_minibatch_loss_average

            #
            # loss logging
            #
            logging["step"].append(step)
            logging["loss"].append(step_loss)

        trainer.summarize_training_progress()

    #
    # save model and logging
    #
    encode.save("./bert_encode.model")
    pooler.save("./bert_pooler.model")
    mask_lm.save("./bert_masked.model")
    sentence_prediction.save("./bert_sentence.model")
    print("Saved model.")

    df = pd.DataFrame(logging)
    df.to_csv("./bert.csv", index=False)
    print("Saved logging.")
    
