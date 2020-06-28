import cntk as C
import cntkx as Cx
import os
import pandas as pd

from cntk.layers import Dense, Embedding

num_classes = 9
num_hidden = 100
num_word = 45044

epoch_size = 10
minibatch_size = 1024
num_samples = 7277


def create_reader(path, is_train):
    return C.io.MinibatchSource(C.io.CTFDeserializer(path, C.io.StreamDefs(
        words=C.io.StreamDef(field="word", shape=num_word, is_sparse=True),
        labels=C.io.StreamDef(field="label", shape=num_classes, is_sparse=True))),
                                randomize=is_train, max_sweeps=C.io.INFINITELY_REPEAT if is_train else 1)


def doc2vec(h):
    h = Embedding(num_hidden)(h)
    h = Cx.sequence.reduce_mean(h)
    h = Dense(num_classes, activation=C.sigmoid)(h)
    return h


if __name__ == "__main__":
    #
    # built-in reader
    #
    train_reader = create_reader("./train_doc2vec_map.txt", is_train=True)

    #
    # input, label and model
    #
    input = C.sequence.input_variable(shape=(num_word,), needs_gradient=True)
    label = C.input_variable(shape=(num_classes,))

    model = doc2vec(input)

    input_map = {input: train_reader.streams.words, label: train_reader.streams.labels}

    #
    # loss function and error metrics
    #
    loss = C.cross_entropy_with_softmax(model, label)
    errs = C.classification_error(model, label)

    #
    # optimizer
    #
    learner = C.adam(model.parameters, lr=0.01, momentum=0.9)
    progress_printer = C.logging.ProgressPrinter(tag="Training")

    trainer = C.Trainer(model, (loss, errs), [learner], [progress_printer])

    C.logging.log_number_of_parameters(model)

    #
    # training
    #
    logging = {"epoch": [], "loss": [], "error": []}
    for epoch in range(epoch_size):
        sample_count = 0
        epoch_loss = 0
        epoch_metric = 0
        while sample_count < num_samples:
            data = train_reader.next_minibatch(min(minibatch_size, num_samples - sample_count), input_map=input_map)

            trainer.train_minibatch(data)

            minibatch_count = data[input].num_sequences
            sample_count += minibatch_count
            epoch_loss += trainer.previous_minibatch_loss_average * minibatch_count
            epoch_metric += trainer.previous_minibatch_evaluation_average * minibatch_count

        #
        # loss and error logging
        #
        logging["epoch"].append(epoch + 1)
        logging["loss"].append(epoch_loss / num_samples)
        logging["error"].append(epoch_metric / num_samples)

        trainer.summarize_training_progress()

    #
    # save model and logging
    #
    model.save("./doc2vec.model")
    print("Saved model.")

    df = pd.DataFrame(logging)
    df.to_csv("./doc2vec.csv", index=False)
    print("Saved logging.")
    
