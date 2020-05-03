import pandas as pd


def xlsx_to_dataset(path):
    df = pd.read_excel(path).drop('Unnamed: 0', 1)
    return df


def concat_xlsx_files_into_data_frame(path_wo_number, number_arr):
    data_frames = []
    for i in number_arr:
        df = xlsx_to_dataset(path_wo_number + str(i) + ".xlsx")
        data_frames.append(df)
    return pd.concat(data_frames)
