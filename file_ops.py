import pandas as pd
import tensorflow as tf
import xlrd


def xlsx_to_dataset(path):
    df = pd.read_excel(path).drop('Unnamed: 0', 1)
    target = df[['reference x', 'reference y']].copy()
    df = df.drop(['reference x', 'reference y'], 1)
    return tf.data.Dataset.from_tensor_slices((df.values, target.values))
