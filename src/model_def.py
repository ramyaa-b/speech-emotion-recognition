"""
Model architecture definition.

Kept separate from app.py so both the training notebook and the inference
app build the *identical* graph before loading weights. We load weights
into a freshly-built model (model.load_weights(...)) rather than
tf.keras.models.load_model(...) on the full saved file, because the
attention block's Lambda layer hits a Keras deserialization restriction
when reconstructing from a saved config. Rebuilding from code sidesteps
that entirely and is just as correct, since it's the exact same
architecture that produced the checkpoint.
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    Input,
    Lambda,
    LSTM,
    MaxPooling1D,
    Multiply,
)
from tensorflow.keras.models import Model


def attention_block(inputs):
    """Learns a soft weighting over time frames instead of averaging them."""
    scores = Dense(1, activation="tanh")(inputs)
    scores = tf.keras.layers.Softmax(axis=1)(scores)
    context = Multiply()([inputs, scores])
    context = Lambda(lambda x: tf.reduce_sum(x, axis=1))(context)
    return context, scores


def build_model(input_shape: tuple, n_classes: int = 8):
    """
    Returns (model, attention_model).

    model: full classifier, input (time_steps, n_features) -> softmax over
        n_classes.
    attention_model: same input, but outputs the per-frame attention
        weights instead of a class prediction — used to visualize which
        parts of a clip the model weighted most heavily.
    """
    inp = Input(shape=input_shape)

    x = Conv1D(128, 5, padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.2)(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(128, 5, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.2)(x)

    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.3)(x)

    context, attn_weights = attention_block(x)

    x = Dense(64, activation="relu")(context)
    x = Dropout(0.3)(x)
    out = Dense(n_classes, activation="softmax")(x)

    model = Model(inputs=inp, outputs=out, name="emotion_cnn_bilstm_attention")
    attn_model = Model(inputs=inp, outputs=attn_weights, name="attention_extractor")
    return model, attn_model
