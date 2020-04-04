import numpy as np
from keras import Input, Model
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, BatchNormalization


class QModel:

    def __init__(self, input_shape=None, num_actions=None):
        """
        Wrapper for the keras model
        Parameters
        ----------
        input_shape: np.ndarray
        num_actions: int
        """
        if input_shape and num_actions:
            self.input_shape = input_shape
            self.num_actions = num_actions
            self.keras_model = self.build_model()
            self.compile_model()
            print(self.keras_model.summary())

    def build_model(self):
        """
        build and return keras Model object
        Parameters
        ----------
        Returns
        -------
        trained_models: Model
        """
        input_layer = Input(shape=self.input_shape)
        bn1 = BatchNormalization()(input_layer)
        conv1 = Conv2D(filters=32, kernel_size=(8, 8), strides=4, activation='relu')(bn1)
        mp1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(filters=64, kernel_size=(4, 4), strides=2, activation='relu')(mp1)
        flattened = Flatten()(conv2)
        hl1 = Dense(128, activation='relu')(flattened)
        output_layer = Dense(self.num_actions, activation='linear')(hl1)

        return Model(input_layer, output_layer)

    def compile_model(self):
        self.keras_model.compile(loss='mse', optimizer='rmsprop')

    def train_on_batch(self, s1, target_q):
        loss = self.keras_model.train_on_batch(x=s1, y=target_q)
        return loss

    def predict_q_values(self, state_batch):
        return self.keras_model.predict(x=state_batch)

    def predict_best_action(self, state_batch):
        q = self.predict_q_values(state_batch)
        return np.argmax(q, 1)[0]

    def save(self, output_path='./'):
        self.keras_model.save(output_path)

