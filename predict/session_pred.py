from config import w2vm, Config
import pandas as pd
from gensim.models import Word2Vec
import os
import sqlalchemy
from sqlalchemy import create_engine, MetaData, select
from sqlalchemy.orm import Session
from sqlalchemy.dialects.sqlite import insert as sqlite_upsert


def load_sessions(dbconnection=Config.dbconnection):
    engine = create_engine(dbconnection)

    metadata = MetaData()
    metadata.reflect(bind=engine)
    session_source = sqlalchemy.Table('session_source', metadata, autoload=True)
    session_dataID_model = sqlalchemy.Table('session_dataID_model', metadata, autoload=True)

    stmt = select(session_source).where(session_source.c.session_dataID.notin_(
            select(session_dataID_model.c.session_dataID)))
    df = pd.read_sql(stmt, engine)  # load sql table into pandas dataframe

    return df


def create_sentences(df):
    session_dataIDs = list(df['session_dataID'].unique())
    sentences = list(df[['sessionID', 'page', 'session_dataID']].groupby(['session_dataID','sessionID']).agg(list)['page'])
    return sentences, session_dataIDs

def train_model(session_model_path, sentences):

    if os.path.exists(session_model_path):
        model = Word2Vec.load(session_model_path)
        model.build_vocab(sentences, update = True)
        model.train(sentences, total_examples=model.corpus_count, epochs=w2vm.epoch)
    else:
        model = Word2Vec(
            sentences=sentences,
            window=w2vm.window,
            vector_size=w2vm.vector_size,
            alpha=w2vm.alpha,
            min_alpha=w2vm.min_alpha,
            min_count=w2vm.min_count,
            negative=w2vm.negative,
        )
        model.save(session_model_path)
    return model


def session_prediction(sentences, model, recnumber, session_dataIDs, dbconnection=Config.dbconnection):
    """This function predicts page IDs for each page and returns a dataframe"""

    flat_list = [x for inner_list in sentences for x in inner_list]
    flat_list = [int(x) for x in flat_list]
    pages_w2v = set(flat_list)

    session_pred_df = pd.DataFrame(columns = ['pageID', 'rec_no', 'session_rec'])

    pageid = []
    recno = []
    rec =[]

    for page in pages_w2v:
        recs = model.wv.most_similar(page, topn=recnumber+1)
        rec_list = [x[0] for x in recs]
        rec_list = rec_list[1:]
        rec.extend(rec_list)
        pageid.extend(recnumber*[page])
        recno.extend(range(1,recnumber+1))

    session_pred_df['pageID'] = pageid
    session_pred_df['rec_no'] = recno
    session_pred_df['session_rec'] = rec

    #update session_dataID_model

    dict_insert = [{'session_dataID': int(value)} for value in session_dataIDs]

    engine = create_engine(dbconnection)
    metadata = MetaData()
    metadata.reflect(bind=engine)
    session_dataID_model = sqlalchemy.Table('session_dataID_model', metadata, autoload=True)

    with Session(engine) as session:
      session.execute(session_dataID_model.insert(), dict_insert)
      session.commit()

    return session_pred_df


def sessionpred_to_db(df, dbconnection):
    """This function updates te predictions in the prediction database"""

    engine = create_engine(dbconnection)

    metadata = MetaData()
    metadata.reflect(bind=engine)
    session_predictions = sqlalchemy.Table('session_predictions', metadata, autoload=True)

    listToWrite = df.to_dict(orient='records')
    stmt = sqlite_upsert(session_predictions).values(listToWrite)
    stmt = stmt.on_conflict_do_update(index_elements=['pageID', 'rec_no'], set_=dict(session_rec=stmt.excluded.session_rec
                                                                                     ))
    #stmt = stmt.on_conflict_nothing(index_elements=['pageID', 'base_dataID']) #activate this (and take the line above out)
    #                                                                         if "old" rows shall not be updated
    with Session(engine) as session:
        session.execute(stmt)
        session.commit()



def session_predict(session_model_path=Config.session_model_path, dbconnection = Config.dbconnection, recnumber = Config.rec_no):
    df = load_sessions()
    sentences, session_dataIDs = create_sentences(df)
    if session_dataIDs:
        model = train_model(session_model_path, sentences)
        session_pred_df = session_prediction(sentences, model, recnumber, session_dataIDs, dbconnection)
        sessionpred_to_db(session_pred_df, dbconnection)

