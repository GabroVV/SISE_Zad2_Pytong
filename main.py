import file_ops as fo

if __name__ == '__main__':
    dataset = fo.xlsx_to_dataset('files/pozyxAPI_dane_pomiarowe/pozyxAPI_only_localization_measurement1.xlsx')
    for feat, targ in dataset.take(5):
        print('Features: {}, Target: {}'.format(feat, targ))
