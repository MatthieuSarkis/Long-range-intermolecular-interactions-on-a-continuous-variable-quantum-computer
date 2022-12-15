import numpy as np
import tensorflow as tf
import strawberryfields as sf

from src.circuit import Circuit

class VQE():

    def __init__(self, modes, layers, active_sd=0.0001, passive_sd=0.1, cutoff_dim=6):

        self.modes = modes
        self.layers = layers
        self.cutoff_dim = cutoff_dim

        self.eng = sf.Engine(backend="tf", backend_options={"cutoff_dim": self.cutoff_dim})
        self.qnn = sf.Program(self.modes)

        self.weights = self.init_weights(active_sd=active_sd, passive_sd=passive_sd) # our TensorFlow weights
        num_params = np.prod(self.weights.shape)   # total number of parameters in our model

        self.sf_params = np.arange(num_params).reshape(self.weights.shape).astype(np.str)
        self.sf_params = np.array([self.qnn.params(*i) for i in self.sf_params])

        self.circuit = Circuit(self.qnn, self.sf_params, self.layers)

        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_history = None
        self.state = None

    def init_weights(self, active_sd=0.0001, passive_sd=0.1):
        """Initialize a 2D TensorFlow Variable containing normally-distributed
        random weights for an ``N`` mode quantum neural network with ``L`` layers.

        Args:
            active_sd (float): the standard deviation used when initializing
                the normally-distributed weights for the active parameters
                (displacement, squeezing, and Kerr magnitude)
            passive_sd (float): the standard deviation used when initializing
                the normally-distributed weights for the passive parameters
                (beamsplitter angles and all gate phases)

        Returns:
            tf.Variable[tf.float32]: A TensorFlow Variable of shape
            ``[layers, 2*(max(1, modes-1) + modes**2 + modes)]``, where the Lth
            row represents the layer parameters for the Lth layer.
        """

        # Number of interferometer parameters:
        M = int(self.modes * (self.modes - 1)) + max(1, self.modes - 1)

        # Create the TensorFlow variables
        int1_weights = tf.random.normal(shape=[self.layers, M], stddev=passive_sd)
        s_weights = tf.random.normal(shape=[self.layers, self.modes], stddev=active_sd)
        int2_weights = tf.random.normal(shape=[self.layers, M], stddev=passive_sd)
        dr_weights = tf.random.normal(shape=[self.layers, self.modes], stddev=active_sd)
        dp_weights = tf.random.normal(shape=[self.layers, self.modes], stddev=passive_sd)
        k_weights = tf.random.normal(shape=[self.layers, self.modes], stddev=active_sd)

        weights = tf.concat(
            [int1_weights, s_weights, int2_weights, dr_weights, dp_weights, k_weights], axis=1
        )

        weights = tf.Variable(weights)

        return weights

    def cost(self, weights):

        mapping = {p.name: w for p, w in zip(self.sf_params.flatten(), tf.reshape(weights, [-1]))}

        state = self.eng.run(self.qnn, args=mapping).state

        x = tf.reshape(tf.stack([state.quad_expectation(mode=i, phi=0.0)[0] for i in range(self.modes)]), shape=(self.modes, 1))
        p = tf.reshape(tf.stack([state.quad_expectation(mode=i, phi=0.5*np.pi)[0] for i in range(self.modes)]), shape=(self.modes, 1))

        gamma = tf.Variable(np.ones(shape=(self.modes, self.modes)) - np.eye(self.modes), dtype=tf.float32)
        H = 0.5 * tf.reduce_sum(x**2 + p**2) + 0.25 * tf.matmul(tf.transpose(x), tf.matmul(gamma, x))
        return H[0][0], state

    def train(self, epochs):

        self.loss_history = []

        for i in range(epochs):

            if self.eng.run_progs:
                self.eng.reset()

            with tf.GradientTape() as tape:
                loss, self.state = self.cost(self.weights)

            self.loss_history.append(loss)

            gradients = tape.gradient(loss, self.weights)

            self.optimizer.apply_gradients(zip([gradients], [self.weights]))

            if i % 1 == 0:
                print("Rep: {} Cost: {:.20f}".format(i, loss))