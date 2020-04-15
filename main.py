import file_ops as fo

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


def build_model():
    _model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    _model.compile(loss='mse',
                   optimizer=optimizer,
                   metrics=['mae', 'mse'])
    return _model


if __name__ == '__main__':
    dataset = fo.xlsx_to_dataset('files/pozyxAPI_dane_pomiarowe/pozyxAPI_only_localization_measurement1.xlsx')
    print(dataset)

    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    train_stats = train_dataset.describe()
    train_stats.pop('reference x')
    train_stats.pop('reference y')
    train_stats = train_stats.transpose()
    print(train_stats)

    train_labels = train_dataset.pop('reference x')
    train_dataset.pop('reference y')
    test_labels = test_dataset.pop('reference x')
    test_dataset.pop('reference y')

    normed_train_data = norm(train_dataset)
    normed_test_data = norm(test_dataset)

    model = build_model()
    print(model.summary())

    example_batch = normed_train_data[:10]
    example_result = model.predict(example_batch)
    print(example_result)

    EPOCHS = 1000

    history = model.fit(
        normed_train_data, train_labels,
        epochs=EPOCHS, validation_split=0.2, verbose=0,
        callbacks=[tfdocs.modeling.EpochDots()])

    hist = fo.pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist.tail())

    plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
    plotter.plot({'Basic': history}, metric="mse")
    plt.ylabel('MAE')
    # for feat, targ in dataset.take(5):
    #     print('Features: {}, Target: {}'.format(feat, targ))
