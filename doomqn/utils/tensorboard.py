
import tensorflow as tf


class TensorboardLogger(object):

    def __init__(self, log_dir):
        """Creates a summary writer logging to log_dir."""
        self.writer = tf.summary.create_file_writer(log_dir)

    def log_scalar(self, scalar_name, value, step):
        """Log a scalar variable.
        Parameter
        ----------
        scalar_name : str
            Name of the scalar
        value
        step : int
            training iteration
        """
        with self.writer.as_default():
            tf.summary.scalar(scalar_name, value, step=step)