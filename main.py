import file_ops as fo
import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling


# def norm(x):
#     return (x - train_stats['mean']) / train_stats['std']


def build_model():
    _model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(2)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    _model.compile(loss='mse',
                   optimizer=optimizer,
                   metrics=['mae', 'mse'])
    return _model


if __name__ == '__main__':
    final_test_data = fo.xlsx_to_dataset('files/pozyxAPI_only_localization_dane_testowe_i_dystrybuanta.xlsx')
    final_test_data_ref = final_test_data.drop(columns=['0/timestamp','t','no','measurement x','measurement y',
                                                        'błąd pomiaru', 'liczba błędnych próbek', '% błędnych próbek',
                                                        'błąd', 'Unnamed: 9'])
    final_test_data_ref = final_test_data_ref.dropna()

    final_test_data_measure = final_test_data.drop(columns=['0/timestamp','t','no','reference x','reference y',
                                                        'błąd pomiaru', 'liczba błędnych próbek', '% błędnych próbek',
                                                        'błąd', 'Unnamed: 9'])
    final_test_data_ref.dropna()
    print(final_test_data_ref)

    dataset = fo.xlsx_to_dataset('files/pozyxAPI_dane_pomiarowe/pozyxAPI_only_localization_measurement1.xlsx')
    dataset.pop('0/timestamp')
    dataset.pop('t')
    dataset.pop('no')

    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)



    train_labels = train_dataset.drop(columns=['measurement x', 'measurement y'])

    test_labels = test_dataset.drop(columns=["measurement x", "measurement y"])


    train_dataset = train_dataset.drop(columns=["reference x", "reference y"])
    test_dataset = test_dataset.drop(columns=["reference x", "reference y"])

    # sns.set()
    # sns.relplot(x="measurement x", y="measurement y",
    #             data=train_dataset)
    # sns.relplot(x="measurement x", y="measurement y",
    #             data=test_dataset)
    # plt.show()

    model = build_model()
    EPOCHS = 1000
    history = model.fit(
        train_dataset, train_labels,
        epochs=EPOCHS, validation_split=0.2, verbose=0,
        callbacks=[tfdocs.modeling.EpochDots()])

    hist = fo.pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist.tail())

    print(history)

    test_predictions = model.predict(test_dataset)

    test_predictions_dataframe = pd.DataFrame(test_predictions, columns=['x', 'y'])


    final_test_data_predictions = model.predict(final_test_data_measure)
    final_test_data_predictions_dataframe = pd.DataFrame(final_test_data_predictions, columns=['x', 'y'])

    sns.relplot(x="x", y="y",
                data= final_test_data_predictions_dataframe)
    sns.relplot(x="reference x", y="reference y",
                data=final_test_data_ref)
    sns.relplot(x="measurement x", y="measurement y",
                data=final_test_data_measure)
    plt.show()

