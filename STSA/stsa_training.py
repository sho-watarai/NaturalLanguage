import cntk as C
import os
import sentencepiece as spm

from cntk.layers import AttentionModel, Dense, Dropout, Embedding, LayerNormalization, LSTM, Recurrence
from cntkx.learners import CyclicalLearningRate
from pandas import DataFrame

num_attention = 512
num_hidden = 512
num_stack = 5
num_word = 32000

epoch_size = 100
minibatch_size = 2048
num_samples = 973124

sample_size = 128
step_size = num_samples // sample_size * 10


class SentencePiece:
    def __init__(self, spm_path):
        self.spm_model = spm.SentencePieceProcessor()
        self.spm_model.Load(spm_path)

    def encode(self, string):
        return self.spm_model.EncodeAsIds(string)

    def decode(self, ids):
        return self.spm_model.DecodeIds(ids)


def create_reader(path, is_train):
    return C.io.MinibatchSource(C.io.CTFDeserializer(path, C.io.StreamDefs(
        tweet=C.io.StreamDef(field="tweet", shape=num_word, is_sparse=True),
        reply=C.io.StreamDef(field="reply", shape=num_word, is_sparse=True))),
                                randomize=is_train, max_sweeps=C.io.INFINITELY_REPEAT if is_train else 1)


def stsa(num_attention, num_hidden, num_stack, num_word):
    embed = Embedding(num_hidden)

    #
    # Encoder
    #
    with C.layers.default_options(enable_self_stabilization=True):
        lstm_forward = [LSTM(num_hidden // 2, init=C.uniform(0.04)) for _ in range(num_stack)]
        lstm_backward = [LSTM(num_hidden // 2, init=C.uniform(0.04)) for _ in range(num_stack)]
        ln_forward = [LayerNormalization() for _ in range(num_stack)]
        ln_backward = [LayerNormalization() for _ in range(num_stack)]
        dropout_enc = [Dropout(0.2) for _ in range(num_stack)]

        @C.Function
        def encoder(input):
            h_enc = embed(input)

            for i in range(num_stack):
                if i == 0:
                    h_enc_forward = ln_forward[i](Recurrence(lstm_forward[i])(h_enc))
                    h_enc_backward = ln_backward[i](Recurrence(lstm_backward[i], go_backwards=True)(h_enc))
                    h_enc = dropout_enc[i](C.splice(h_enc_forward, h_enc_backward))
                else:
                    h_enc_forward = ln_forward[i](Recurrence(lstm_forward[i])(h_enc))
                    h_enc_backward = ln_backward[i](Recurrence(lstm_backward[i], go_backwards=True)(h_enc))
                    h_enc = dropout_enc[i](C.splice(h_enc_forward, h_enc_backward)) + h_enc

            return h_enc

    #
    # Decoder
    #
    with C.layers.default_options(enable_self_stabilization=True):
        attention = AttentionModel(num_attention, init=C.uniform(0.04), name="attention")
        lstm = [LSTM(num_hidden, init=C.uniform(0.04)) for _ in range(num_stack)]
        dropout_dec = [Dropout(0.2) for _ in range(num_stack)]
        ln_dec = [LayerNormalization() for _ in range(num_stack)]
        dense = Dense(num_word, init=C.uniform(0.04))

        @C.Function
        def decoder(input, label):
            h_dec = embed(label)

            for i in range(num_stack):
                if i == 0:
                    @C.Function
                    def lstm_with_attention(h, c, x):
                        attention_encoded = attention(encoder(input), h)
                        return lstm[i](h, c, C.splice(attention_encoded, x))
                    h_dec = dropout_dec[i](ln_dec[i](Recurrence(lstm_with_attention)(h_dec)))
                else:
                    h_dec = dropout_dec[i](ln_dec[i](Recurrence(lstm[i])(h_dec))) + h_dec

            return dense(h_dec)

    return decoder


def write_text(model, data, encode, decode, spm_model):
    sources = encode * 1
    queries = sources.eval({sources.arguments[0]: data[encode].data})
    predicts = model.eval({encode: data[encode].data, decode: data[decode].data})

    for query, predict in zip(queries, predicts):
        tweet = spm_model.decode([int(q) for q in query.argmax(axis=1)])
        print(tweet, end=" -> ")

        reply = spm_model.decode([int(p) for p in predict.argmax(axis=1)])
        print(reply)

    return tweet, reply


if __name__ == "__main__":
    #
    # Sentence Piece model
    #
    spm_model = SentencePiece("./twitter.model")

    #
    # built-in reader
    #
    train_reader = create_reader("./tweet_reply_corpus.txt", is_train=True)

    #
    # tweet, reply and model
    #
    tweet = C.sequence.input_variable(shape=(num_word,), needs_gradient=True, sequence_axis=C.Axis("E"))
    reply = C.sequence.input_variable(shape=(num_word,), needs_gradient=True, sequence_axis=C.Axis("D"))

    model = stsa(num_attention, num_hidden, num_stack, num_word)(tweet, C.sequence.slice(reply, 0, -1))

    input_map = {tweet: train_reader.streams.tweet, reply: train_reader.streams.reply}

    #
    # loss function and perplexity metrics
    #
    loss = C.cross_entropy_with_softmax(model, C.sequence.slice(reply, 1, 0))
    ppl = C.exp(loss)

    #
    # optimizer and cyclical learning rate
    #
    learner = C.adam(model.parameters, lr=0.1, momentum=0.9,
                     gradient_clipping_threshold_per_sample=sample_size, gradient_clipping_with_truncation=True)
    clr = CyclicalLearningRate(learner, base_lr=1e-4, max_lr=0.1, ramp_up_step_size=step_size, gamma=0.99994,
                               minibatch_size=sample_size, lr_policy="exp_range")
    progress_printer = C.logging.ProgressPrinter(tag="Training")

    trainer = C.Trainer(model, (loss, ppl), [learner], [progress_printer])

    C.logging.log_number_of_parameters(model)
    
    #
    # training
    #
    logging = {"epoch": [], "loss": [], "ppl": [], "tweet": [], "reply": []}
    for epoch in range(epoch_size):
        sample_count = 0
        epoch_loss = 0
        epoch_metric = 0
        while sample_count < num_samples:
            data = train_reader.next_minibatch(min(minibatch_size, num_samples - sample_count), input_map=input_map)

            trainer.train_minibatch(data)
            
            clr.batch_step()

            minibatch_count = data[tweet].num_sequences
            sample_count += minibatch_count
            epoch_loss += trainer.previous_minibatch_loss_average * minibatch_count
            epoch_metric += trainer.previous_minibatch_evaluation_average * minibatch_count

        twt, rep = write_text(model, data, tweet, reply, spm_model)

        #
        # loss and error logging
        #
        logging["epoch"].append(epoch + 1)
        logging["loss"].append(epoch_loss / num_samples)
        logging["ppl"].append(epoch_metric / num_samples)
        logging["tweet"].append(twt)
        logging["reply"].append(rep)

        trainer.summarize_training_progress()

    #
    # save model and logging
    #
    model.save("./stsa.model")
    print("Saved model.")

    df = DataFrame(logging)
    df.to_csv("./stsa.csv", index=False)
    print("Saved logging.")
    
