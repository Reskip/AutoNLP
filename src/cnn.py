from tensorflow.python.keras import regularizers, Input, Model
from tensorflow.python.keras.initializers import Constant
from tensorflow.python.keras.layers import Activation, BatchNormalization,\
    concatenate, Convolution1D, Dropout, Dense, Embedding, Flatten, MaxPooling1D


def cnn_model(
    input_shape,        # input list shape (word index list)
    num_classes,        # output shape
    num_features,       # number of word + empty
    embedding_matrix,
    filters,
    kernel_sizes,
    dropout_rate,
    embedding_trainable,
    l2_lambda
):
    embedding_layer = Embedding(
        input_dim=num_features,
        output_dim=300,  # hard code
        embeddings_initializer=Constant(embedding_matrix),
        input_length=input_shape,
        trainable=embedding_trainable)

    # word index list, not map to embedding yet
    sequence_input = Input(shape=(input_shape,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    nn_layers = list()
    for kernel_size in kernel_sizes:
        conv_layer_0 = Convolution1D(
            filters, kernel_size, padding='valid')(embedded_sequences)
        conv_layer_1 = BatchNormalization(axis=1)(conv_layer_0)
        conv_layer_2 = Activation('relu')(conv_layer_1)
        pool_layer_0 = MaxPooling1D(
            input_shape - kernel_size + 1)(conv_layer_2)
        pool_layer_1 = Dropout(dropout_rate)(pool_layer_0)

        nn_layers.append(pool_layer_1)

    # merge diff kernal size generated output
    line_merge_layer = concatenate(nn_layers)
    line_flat_layer = Flatten()(line_merge_layer)

    norm_layer = BatchNormalization(axis=1)(line_flat_layer)
    drop_layer = Dropout(dropout_rate)(norm_layer)

    preds = Dense(num_classes,
                  kernel_regularizer=regularizers.l2(l2_lambda),
                  activation='softmax')(drop_layer)

    cnn_model = Model(inputs=sequence_input, outputs=preds)

    return cnn_model
