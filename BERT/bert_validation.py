import cntk as C

num_classes = 9
num_segment = 2
num_word = 32000

minibatch_size = 1
num_samples = 90


def create_reader(path, is_train):
    return C.io.MinibatchSource(C.io.CTFDeserializer(path, C.io.StreamDefs(
        token=C.io.StreamDef(field="word", shape=num_word, is_sparse=True),
        segment=C.io.StreamDef(field="segment", shape=num_segment, is_sparse=True),
        label=C.io.StreamDef(field="label", shape=num_classes, is_sparse=True))),
                                randomize=is_train, max_sweeps=C.io.INFINITELY_REPEAT if is_train else 1)


if __name__ == "__main__":
    #
    # built-in reader
    #
    valid_reader = create_reader("./val_doc2vec_map.txt", is_train=False)

    #
    # model and label
    #
    model = C.load_model("./bert_finetuning.model")
    label = C.input_variable(shape=(num_classes,), is_sparse=True)

    input_map = {model.arguments[0]: valid_reader.streams.token, model.arguments[1]: valid_reader.streams.segment,
                 label: valid_reader.streams.label}

    #
    # loss function and error metrics
    #
    errs = C.classification_error(model, label)

    #
    # validation
    #
    sample_count = 0
    error = 0
    while sample_count < num_samples:
        data = valid_reader.next_minibatch(min(minibatch_size, num_samples - sample_count), input_map=input_map)

        error += errs.eval(data).sum()

        sample_count += data[label].num_sequences

    print("Accuracy {:.2f}%".format((num_samples - error) / num_samples * 100))
    
