import cntk as C
import numpy as np
import os

from cntk.layers import Embedding
from cntkx.learners import CyclicalLearningRate
from pandas import DataFrame

is_CBOW = False

num_hidden = 100
num_word = 6253
num_window = 5

sample_size = 5
sampling_weights = np.power(np.load("./sampling_weights.npy"), 0.75).reshape(1, num_word)
allow_duplicates = False

epoch_size = 10
minibatch_size = 128
if is_CBOW:
    num_samples = 28164  # CBOW
else:
    num_samples = 281640  # Skip-gram

step_size = num_samples // minibatch_size * 2


def create_reader(path, is_train):
    return C.io.MinibatchSource(C.io.CTFDeserializer(path, C.io.StreamDefs(
        words=C.io.StreamDef(field="word", shape=num_word, is_sparse=True),
        targets=C.io.StreamDef(field="target", shape=num_word, is_sparse=True))),
                                randomize=is_train, max_sweeps=C.io.INFINITELY_REPEAT if is_train else 1)


def sampled_softmax(hidden_vector, target_vector, num_word, num_hidden, sample_size, sampling_weights):
    W = C.parameter(shape=(num_word, num_hidden), init=C.glorot_uniform())

    sample_selector = C.random_sample(sampling_weights, sample_size, allow_duplicates)
    inclusion_probs = C.random_sample_inclusion_frequency(sampling_weights, sample_size, allow_duplicates)
    log_prior = C.log(inclusion_probs)

    W_sampled = C.times(sample_selector, W)
    z_sampled = C.times_transpose(W_sampled, hidden_vector) - C.times_transpose(sample_selector, log_prior)

    W_target = C.times(target_vector, W)
    z_target = C.times_transpose(W_target, hidden_vector) - C.times_transpose(target_vector, log_prior)

    z_reduced = C.reduce_log_sum_exp(z_sampled)

    loss = C.log_add_exp(z_target, z_reduced) - z_target

    z = C.times_transpose(W, hidden_vector)
    z = C.reshape(z, shape=num_word)

    zSMax = C.reduce_max(z_sampled)
    errs = C.less(z_target, zSMax)

    return z, loss, errs


if __name__ == "__main__":
    #
    # built-in reader
    #
    if is_CBOW:
        train_reader = create_reader("./corpus_cbow.txt", is_train=True)
    else:
        train_reader = create_reader("./corpus_skipgram.txt", is_train=True)

    #
    # input, label and embed
    #
    if is_CBOW:
        input = C.sequence.input_variable(shape=(num_word,))
        label = C.input_variable(shape=(num_word,), is_sparse=True)
        embed = C.sequence.reduce_sum(Embedding(num_hidden)(input)) / (num_window * 2)
    else:
        input = C.input_variable(shape=(num_word,))
        label = C.input_variable(shape=(num_word,), is_sparse=True)
        embed = Embedding(num_hidden)(input)

    input_map = {input: train_reader.streams.words, label: train_reader.streams.targets}

    #
    # loss function and error metrics
    #
    model, loss, errs = sampled_softmax(embed, label, num_word, num_hidden, sample_size, sampling_weights)

    #
    # training setting
    #
    learner = C.adam(model.parameters, lr=0.01, momentum=0.9)
    clr = CyclicalLearningRate(learner, base_lrs=1e-4, max_lrs=0.01, minibatch_size=minibatch_size, step_size=step_size)
    progress_printer = C.logging.ProgressPrinter(tag="Training")

    trainer = C.Trainer(model, (loss, errs), [learner], [progress_printer])

    C.logging.log_number_of_parameters(model)

    logging = {"epoch": [], "loss": [], "error": []}
    for epoch in range(epoch_size):
        sample_count = 0
        epoch_loss = 0
        epoch_metric = 0
        while sample_count < num_samples:
            data = train_reader.next_minibatch(min(minibatch_size, num_samples - sample_count), input_map=input_map)

            trainer.train_minibatch(data)

            clr.batch_step()

            sample_count += minibatch_size
            epoch_loss += trainer.previous_minibatch_loss_average
            epoch_metric += trainer.previous_minibatch_evaluation_average

        #
        # loss and error logging
        #
        logging["epoch"].append(epoch + 1)
        logging["loss"].append(epoch_loss / (num_samples / minibatch_size))
        logging["error"].append(epoch_metric / (num_samples / minibatch_size))

        trainer.summarize_training_progress()

    #
    # save model and logging
    #
    if is_CBOW:
        model.save("./cbow.model")
    else:
        model.save("./skipgram.model")
    print("Saved model.")

    df = DataFrame(logging)
    if is_CBOW:
        df.to_csv("./cbow.csv", index=False)
    else:
        df.to_csv("./skipgram.csv", index=False)
    print("Saved logging.")
    
