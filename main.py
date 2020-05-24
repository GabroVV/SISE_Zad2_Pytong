import file_ops as fo
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling


def build_model():
    initializer = initializers.RandomNormal(mean=0.0, stddev=0.05)
    _model = keras.Sequential([
        layers.Dense(32, activation='relu', input_shape=[2, ],
                     kernel_initializer=initializer),
        layers.Dense(64, activation='relu', kernel_initializer=initializer),
        layers.Dense(64, activation='relu', kernel_initializer=initializer),
        layers.Dense(2, activation='linear', kernel_initializer=initializer)
    ])

    _model.compile(loss='mse',
                   optimizer='adam',

                   metrics=['mae', 'mse'])
    return _model


if __name__ == '__main__':
    final_test_data = fo.xlsx_to_dataset('files/pozyxAPI_only_localization_dane_testowe_i_dystrybuanta.xlsx')
    final_test_data = final_test_data.drop(columns=['0/timestamp', 't', 'no',
                                                    'błąd pomiaru', 'liczba błędnych próbek', '% błędnych próbek',
                                                    'błąd', 'Unnamed: 9'])

    final_test_data_ref = final_test_data.drop(
        columns=['measurement x', 'measurement y'])  # reference x, reference y left
    final_test_data_ref = final_test_data_ref.dropna()

    final_test_data_measure = final_test_data.drop(
        columns=['reference x', 'reference y'])  # measurement x, measurement y left
    final_test_data_ref.dropna()

    file_numbers = [*range(1, 13, 1)]
    train_dataset = fo.concat_xlsx_files_into_data_frame(
        'files/pozyxAPI_dane_pomiarowe/pozyxAPI_only_localization_measurement', file_numbers)
    train_dataset = train_dataset.drop(columns=['0/timestamp', 't', 'no'])

    train_labels = train_dataset.drop(columns=['measurement x', 'measurement y'])
    train_dataset = train_dataset.drop(columns=["reference x", "reference y"])

    model = build_model()
    model.summary()
    EPOCHS = 500
    history = model.fit(
        train_dataset, train_labels,
        epochs=EPOCHS, validation_split=0.1, verbose=0,
        callbacks=[tfdocs.modeling.EpochDots()])

    hist = fo.pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print("\n")
    print(hist.tail())

    final_test_data_predictions = model.predict(final_test_data_measure)
    final_test_data_predictions_data_frame = pd.DataFrame(final_test_data_predictions, columns=['x', 'y'])

    sns.relplot(x="x", y="y",
                data=final_test_data_predictions_data_frame)
    sns.relplot(x="reference x", y="reference y",
                data=final_test_data_ref)
    sns.relplot(x="measurement x", y="measurement y",
                data=final_test_data_measure)
    plt.show()
    final_test_data_predictions_data_frame.to_clipboard(excel=True)
