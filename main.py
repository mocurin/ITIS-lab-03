"""App entry point"""
import logging
import sys

import matplotlib.pyplot as plt

from sources.metrics import mean_square_error
from sources.models import Neuron
from sources.generators import window_at_position, function_as_points, DataGenerator


logging.basicConfig(level=logging.INFO, stream=sys.stdout)


FUNCTION = lambda t: (t ** 4) - 2 * (t ** 3) + t
DEFAULT_WINDOW_SIZE = 4
RANGE = (-.5, .5)
POINTS = 20
DELTA = 0.000001
MAX_EPOCH = 10000
NORM = 0.033
SEED = 42


def create_train_data(values, window_size: int):
    def _generator():
        while True:
            for idx, _ in enumerate(values[window_size:]):
                # We have to shift `idx` as `idx` in `window_at_position`
                # represents rightmost window pos + 1
                idx += window_size
                yield window_at_position(values, window_size, idx)

    return len(values) - window_size, _generator()


def create_and_fit(values, window_size: int):
    # Create train & validation data
    epoch_size, train_data = create_train_data(values, window_size)
    validation_data = DataGenerator(train_data, epoch_size)
    train_data = DataGenerator(train_data, epoch_size, MAX_EPOCH)

    # Create model to fit
    model = Neuron(
        window_size,
        use_bias=True,
        metrics={'MSE': mean_square_error},
        # metrics_early_stop={'MSE': lambda x: x < DELTA}
    )

    # Fit, return model itself
    return model.fit(
        train_data,
        validation_data,
        NORM
    ), model


def take_and_evaluate(model, values, window_size: int):
    # Lets copy list -> generators wont break after appends
    source = [value for value in values]

    for idx in range(POINTS, POINTS * 2):
        window, _ = window_at_position(source, window_size, idx)

        # Predict result
        output = model.predict(window)

        # Extend list with new value
        source.append(output)

    return source


def main():
    left, right = RANGE

    # y for (a, b)
    values = function_as_points(FUNCTION, RANGE, POINTS)

    # y for (a, 2b-a)
    y_values = function_as_points(FUNCTION, (left, 2 * right - left), POINTS * 2)

    # t for (a, 2b-a)
    t_values = function_as_points(lambda x: x, (left, 2 * right - left), POINTS * 2)

    for window_size in range(DEFAULT_WINDOW_SIZE, POINTS):
        logging.info(f"Started fitting. Window size: {window_size}")

        # Fit neuron
        (history, event), model = create_and_fit(values, window_size)
        logging.info(f"Fitting completed. Weights: {model._weights}")

        # Predict values for (b, 2b-a)
        prediction = take_and_evaluate(model, values, window_size)

        # Create plot for current window size
        plt.title(f"Function: y(t) = t^4 - 2t^3 + t. Window size: {window_size}")
        plt.plot(t_values, prediction, marker='+')
        plt.plot(t_values, y_values)
        plt.axvline(RANGE[1], c='r')
        plt.savefig(f"ws{window_size}.png")
        # plt.show()
        plt.clf()


if __name__ == '__main__':
    main()
