import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import declarative_base, Session, sessionmaker
from config import Config
from sqlalchemy.dialects.sqlite import insert as sqlite_upsert


def load_session_data(sessionfile_path):
    """This function loads the sessions file into memory and does basic cleanup
      as well as some transformations to prepare it for the database"""

    sessions_df_raw = pd.read_excel(sessionfile_path, sheet_name='Sessions')
    sessions_df_raw = sessions_df_raw.rename(
                     columns={'Data ID': 'session_dataID', 'sessionId': 'sessionID', 'User ID': 'userID'})



    sessions_df = pd.DataFrame(columns = ['session_dataID', 'sessionID', 'order', 'page', 'userID'])

    #transform df to target form prescribed by tha database
    sdid = []
    sid = []
    order = []
    page = []
    user = []

    for d in sessions_df_raw['session_dataID'].unique():

        df_temp = sessions_df_raw[sessions_df_raw['session_dataID']==d].copy()

        sessions = df_temp.iloc[:, 3:].values.tolist()
        sessions = [list(filter(lambda x: x == x, inner_list)) for inner_list in sessions]
        sessions = [[int(x) for x in inner_list] for inner_list in sessions]

        sessionIDs = list(df_temp['sessionID'])
        userIDs = list(df_temp['userID'])

        for s in sessions:
            page.extend(s)
            sid.extend(len(s)*[sessionIDs[sessions.index(s)]])
            order.extend(range(1,len(s)+1))
            sdid.extend(len(s)*[d])
            user.extend(len(s)*[userIDs[sessions.index(s)]])

        sessions_df['session_dataID'] = sdid
        sessions_df['sessionID'] = sid
        sessions_df['order'] = order
        sessions_df['page'] = page
        sessions_df['userID'] = user

    return sessions_df

def update_session_source(df):

    engine = create_engine(Config.dbconnection)

    metadata = MetaData()
    metadata.reflect(bind=engine)
    session_source = sqlalchemy.Table('session_source', metadata, autoload = True)

    listToWrite = df.to_dict(orient='records')
    stmt = sqlite_upsert(session_source).values(listToWrite)
    stmt = stmt.on_conflict_do_update(index_elements=['session_dataID', 'sessionID', 'order'],
                                      set_=dict(page=stmt.excluded.page
                                                ))
    # stmt = stmt.on_conflict_nothing(index_elements=['session_dataID', 'sessionID', 'order']) #activate this (and take the line above out)
    #                                                                         if "old" rows shall not be updated

    with Session(engine) as session:
      session.execute(stmt)
      session.commit()


def session_to_db(sessionfile_path = Config.sessionfile_path):
    sessions_df = load_session_data(sessionfile_path)
    update_session_source(sessions_df)