class Config(object):
    db_path = 'kow.db'
    basefile_path = 'kow.xlsx'
    sessionfile_path = 'kow.xlsx'
    session_model_path = 'kow_recommender'
    dbconnection = 'sqlite:///kowrec.db'
    model_name_orig = 'glove-wiki-gigaword-50'
    model_name = 'glove-wiki-gigaword-50'
    rec_no = 5
    num_features = 50


class w2vm:
    window = 5
    vector_size = 20
    alpha = 0.04
    min_alpha = 0.01
    min_count = 1
    negative = 20
    epoch = 5