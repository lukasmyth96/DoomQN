import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Reshape, Dense, MaxPooling2D, Concatenate


class ICM:

    def __init__(self, input_shape, num_actions):
        """
        Intrinsic Curiosity Module from https://arxiv.org/pdf/1705.05363.pdf
        Parameters
        ----------
        input_shape: tuple
        num_actions: int
        """
        self.input_shape = input_shape
        self.num_actions = num_actions

        self.feature_dim = self.compute_feature_dimension()
        self.keras_model = self.build_model()
        self.compile_model()

        print(self.keras_model.summary())

    def build_model(self):
        """
        Builds and returns three keras Models
        Returns
        -------
        keras_model: Model
        """

        s_curr_input = Input(shape=self.input_shape)
        s_next_input = Input(shape=self.input_shape)
        action_input = Input(shape=(self.num_actions,))
        s_curr_features = self.cnn(s_curr_input)
        s_next_features = self.cnn(s_next_input)

        # Inverse Dynamics Model
        inverse_concat_features = Concatenate()([s_curr_features, s_next_features])
        inverse_hl1 = Dense(128, activation='relu')(inverse_concat_features)
        inverse_output = Dense(self.num_actions, activation='softmax', name='inverse_model')(inverse_hl1)

        # Forward Dynamics Model
        forward_concat_features = Concatenate()([s_curr_features, action_input])
        forward_hl1 = Dense(128, activation='relu')(forward_concat_features)
        forward_out = Dense(self.feature_dim, activation='linear', name='forward_model')(forward_hl1)

        keras_model = Model(inputs=[s_curr_input, action_input, s_next_input], outputs=[inverse_output, forward_out])

        return keras_model

    @staticmethod
    def cnn(x):
        x = Conv2D(filters=32, kernel_size=(8, 8), strides=4, activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(filters=64, kernel_size=(4, 4), strides=2, activation='relu')(x)
        flat_dim = np.prod(x.shape.as_list()[1:])  # workaround for issue with Flatten and Reshape
        x = Reshape((flat_dim,))(x)
        return x

    def compute_feature_dimension(self):
        """
        Compute the output dimension for the forward model based on the input shape
        Returns
        -------
        dim: int
        """
        shape = (1,) + self.input_shape
        x = tf.zeros(shape)
        features = self.cnn(x)
        return features.shape[1]

    def compile_model(self):

        losses = {'inverse_model': 'categorical_crossentropy',
                  'forward_model': 'mse'}

        loss_weights = {'inverse_model': 1.0,
                        'forward_model': 1.0}

        self.keras_model.compile(loss=losses, loss_weights=loss_weights, optimizer='rmsprop')

    def train_on_batch(self, batch):
        """
        Parameters
        ----------
        batch: list[np.ndarray]
            [s_curr_input, action_input, s_next_input]

        Returns
        -------
        prediction_error: float
            the prediction error of the forward dynamics model - this is what we'll use as the intrinsic reward
        """
        assert len(batch) == 3
        batch = self.add_batch_dims(batch)
        s_next_input = tf.cast(tf.convert_to_tensor(batch[2]), 'float32')
        s_next_features = self.cnn(s_next_input)  # needed as target
        losses = self.keras_model.train_on_batch(batch, [batch[1], s_next_features])
        prediction_error = losses[2]  # TODO be careful of this
        return prediction_error

    @staticmethod
    def add_batch_dims(batch):
        """
        Add batch dim to each input array to train_on_batch()
        Parameters
        ----------
        batch: list[np.ndarray]
            [current_state, action, next_state]

        Returns
        -------
        new_batch: list[np.ndarray]
        """
        new_batch = []
        for arr in batch:
            new_batch.append(np.expand_dims(arr, axis=0))
        return new_batch


if __name__ == '__main__':
    """ Testing """
    model = ICM(input_shape=(64, 64, 1), num_actions=3)
    s1 = np.random.random((64, 64, 1))
    s2 = np.random.random((64, 64, 1))
    action = np.array([1, 0, 0])
    _batch = [s1, action, s2]
    losses = model.train_on_batch(_batch)
    print('stop here')
