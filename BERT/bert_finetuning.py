import cntk as C
import pandas as pd

from cntk.layers import Dense
from cntkx.learners import CyclicalLearningRate

num_classes = 9
num_segment = 2
num_word = 32000

epoch_size = 5
max_seq_len = 128
minibatch_size = 1024
num_samples = 7277

step_size = num_samples // 8


def create_reader(path, is_train):
    return C.io.MinibatchSource(C.io.CTFDeserializer(path, C.io.StreamDefs(
        token=C.io.StreamDef(field="word", shape=num_word, is_sparse=True),
        segment=C.io.StreamDef(field="segment", shape=num_segment, is_sparse=True),
        label=C.io.StreamDef(field="label", shape=num_classes, is_sparse=True))),
                                randomize=is_train, max_sweeps=C.io.INFINITELY_REPEAT if is_train else 1)


def bert_document_classification(pooler):
    """ document classification """
    h = C.sequence.unpack(pooler, padding_value=0, no_mask_output=True)[0, :]
    return Dense(num_classes, activation=C.sigmoid, init=C.normal(0.02))(h)


if __name__ == "__main__":
    #
    # built-in reader
    #
    train_reader = create_reader("./train_doc2vec_map.txt", is_train=True)

    #
    # encode and pooler
    #
    encode = C.load_model("./bert_encode.model")
    pooler = C.load_model("./bert_pooler.model")

    #
    # token, segment, label, and model
    #
    token = C.sequence.input_variable(shape=(num_word,))
    segment = C.sequence.input_variable(shape=(num_segment,))
    label = C.input_variable(shape=(num_classes,))

    model = bert_document_classification(pooler)

    #
    # loss function and metric error
    #
    loss = C.cross_entropy_with_softmax(model, label)
    errs = C.classification_error(model, label)

    #
    # optimizer and cyclical learning rate
    #
    learner = C.adam(model.parameters, lr=C.learning_parameter_schedule_per_sample(0.1), momentum=0.9,
                     gradient_clipping_threshold_per_sample=minibatch_size, gradient_clipping_with_truncation=True)
    clr = CyclicalLearningRate(learner, base_lr=1e-6, max_lr=1e-4,
                               ramp_up_step_size=step_size, minibatch_size=minibatch_size)
    progress_printer = C.logging.ProgressPrinter(tag="Training")

    trainer = C.Trainer(model, (loss, errs), [learner], [progress_printer])

    input_map = {model.arguments[0]: train_reader.streams.token, model.arguments[1]: train_reader.streams.segment,
                 label: train_reader.streams.label}

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

            clr.batch_step()

            minibatch_count = data[model.arguments[0]].num_sequences
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
    model.save("./bert_finetuning.model")
    print("Saved model.")

    df = pd.DataFrame(logging)
    df.to_csv("./bert_finetuning.csv", index=False)
    print("Saved logging.")
    
