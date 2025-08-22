import tensorflow as tf
import tensorflow_probability as tfp
from config import PRIOR_SCALE, DROPOUT_RATE
tfd = tfp.distributions
tfpl = tfp.layers


def prior_dist_fn(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfpl.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=tf.zeros(n, dtype=dtype),
                       scale=PRIOR_SCALE * tf.ones(n, dtype=dtype)),
            reinterpreted_batch_ndims=1))
    ])


def posterior_dist_fn(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfpl.VariableLayer(2 * n, dtype=dtype),
        tfpl.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t[..., :n],
                       scale=1e-2 + tf.nn.softplus(t[..., n:])),
            reinterpreted_batch_ndims=1))
    ])


def nll_loss():
    def loss_fn(y, rv_y):
        nll = -rv_y.log_prob(y)
        return tf.reduce_mean(nll)
    return loss_fn


def build_bayesian_pm25(input_dim_phi, input_dim_cov, kl_weight=None):
    """
    Builds a Bayesian neural network for PM2.5 prediction.

    The model takes two inputs:
    - Spatial representation (basis functions)
    - Meteorological covariates

    It outputs the mean and standard deviation of a Normal distribution representing PM2.5 concentrations.
    """

    if kl_weight is None:
        raise ValueError(
            "Missing 'kl_weight'. Bayesian models require this for KL divergence regularization.")

    phi_input = tf.keras.Input(shape=(input_dim_phi,), name='phi_input')
    cov_input = tf.keras.Input(shape=(input_dim_cov,), name='cov_input')

    # Spatial branch
    phi_dense = tf.keras.layers.Dense(
        100, activation='relu', kernel_initializer='he_uniform')(phi_input)
    phi_dense = tf.keras.layers.Dropout(
        rate=DROPOUT_RATE)(phi_dense)

    phi_dense = tfpl.DenseVariational(
        100, make_posterior_fn=posterior_dist_fn, make_prior_fn=prior_dist_fn, kl_weight=kl_weight, activation='relu')(phi_dense)

    phi_dense = tf.keras.layers.Dense(100, activation='relu')(
        phi_dense)
    phi_dense = tf.keras.layers.Dropout(
        rate=DROPOUT_RATE)(phi_dense)

    # Covariate branch
    cov_dense = tf.keras.layers.Dense(
        100, activation='relu', kernel_initializer='he_uniform')(cov_input)
    cov_dense = tf.keras.layers.Dropout(
        rate=DROPOUT_RATE)(cov_dense)

    # merge
    merged = tf.keras.layers.Concatenate()([phi_dense, cov_dense])
    merged = tf.keras.layers.Dense(128, activation='relu')(merged)
    merged = tf.keras.layers.Dropout(rate=DROPOUT_RATE)(merged)
    merged = tf.keras.layers.Dense(128, activation='relu')(merged)
    merged = tf.keras.layers.Dropout(rate=DROPOUT_RATE)(merged)
    merged = tf.keras.layers.Dense(100, activation='relu')(merged)
    merged = tf.keras.layers.Dropout(rate=DROPOUT_RATE)(merged)
    merged = tf.keras.layers.Dense(100, activation='relu')(merged)
    merged = tf.keras.layers.Dropout(rate=DROPOUT_RATE)(merged)

    merged = tfpl.DenseVariational(
        100, make_posterior_fn=posterior_dist_fn, make_prior_fn=prior_dist_fn, kl_weight=kl_weight, activation='relu')(merged)

    merged = tf.keras.layers.Dense(32, activation='relu')(merged)
    merged = tf.keras.layers.Dropout(rate=DROPOUT_RATE)(merged)

    # LogNormal distribution output
    loc = tf.keras.layers.Dense(1, name='loc_output')(merged)

    scale = tf.keras.layers.Dense(
        1, activation='softplus', name='scale_output')(merged)
    scale = tf.keras.layers.Lambda(lambda x: x + 1e-3)(scale)

    output = tfpl.DistributionLambda(
        lambda t: tfd.LogNormal(loc=t[0], scale=t[1])
    )([loc, scale])

    return tf.keras.Model(inputs=[cov_input, phi_input], outputs=output)
