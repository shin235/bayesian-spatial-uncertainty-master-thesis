import tensorflow as tf
from config import DROPOUT_RATE


def build_deterministic(input_dim_phi, input_dim_cov):
    """
    Deterministic DeepKriging model for PM2.5 prediction (point estimate only).
    """

    phi_input = tf.keras.Input(shape=(input_dim_phi,), name='phi_input')
    cov_input = tf.keras.Input(shape=(input_dim_cov,), name='cov_input')

    # Spatial branch
    phi_dense = tf.keras.layers.Dense(100, activation='relu')(phi_input)
    phi_dense = tf.keras.layers.Dropout(rate=DROPOUT_RATE)(phi_dense)
    phi_dense = tf.keras.layers.Dense(100, activation='relu')(phi_dense)
    phi_dense = tf.keras.layers.Dropout(rate=DROPOUT_RATE)(phi_dense)

    # Covariate branch
    cov_dense = tf.keras.layers.Dense(100, activation='relu')(cov_input)
    cov_dense = tf.keras.layers.Dropout(rate=DROPOUT_RATE)(cov_dense)

    # Concatenate
    merged = tf.keras.layers.Concatenate()([phi_dense, cov_dense])
    merged = tf.keras.layers.Dense(128, activation='relu')(merged)
    merged = tf.keras.layers.Dropout(rate=DROPOUT_RATE)(merged)
    merged = tf.keras.layers.Dense(128, activation='relu')(merged)
    merged = tf.keras.layers.Dropout(rate=DROPOUT_RATE)(merged)
    merged = tf.keras.layers.Dense(100, activation='relu')(merged)
    merged = tf.keras.layers.Dropout(rate=DROPOUT_RATE)(merged)
    merged = tf.keras.layers.Dense(100, activation='relu')(merged)
    merged = tf.keras.layers.Dropout(rate=DROPOUT_RATE)(merged)
    merged = tf.keras.layers.Dense(32, activation='relu')(merged)

    # Output: point prediction only
    output = tf.keras.layers.Dense(1)(merged)

    return tf.keras.Model(inputs=[cov_input, phi_input], outputs=output)
