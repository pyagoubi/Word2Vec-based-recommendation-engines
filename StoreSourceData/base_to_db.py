import pandas as pd
import nltk
import sqlalchemy
from sqlalchemy import create_engine,  MetaData
from sqlalchemy.orm import  Session
from config import Config
nltk.download('stopwords')
from sqlalchemy.dialects.sqlite import insert as sqlite_upsert



def load_data(basefile_path):
    """This function loads the pages file into memory and does basic cleanup to prepare it for loading into the database"""

    data_pages = pd.read_excel(basefile_path, sheet_name='Pages')
    data_pages = data_pages[['Data ID', 'Page ID', 'Page Title', 'Tags']].rename(columns={'Data ID': 'base_dataID', 'Page ID': 'pageID', 'Page Title': 'page_title', 'Tags': 'tags'})
    data_pages = data_pages.drop(0)
    return data_pages

def update_base(df):
    engine = create_engine(Config.dbconnection)

    metadata = MetaData()
    metadata.reflect(bind=engine)
    base_source = sqlalchemy.Table('base_source', metadata, autoload=True)

    listToWrite = df.to_dict(orient='records')
    stmt = sqlite_upsert(base_source).values(listToWrite)
    stmt = stmt.on_conflict_do_update(index_elements=['pageID', 'base_dataID'], set_=dict(page_title=stmt.excluded.page_title,
                                                                                      tags=stmt.excluded.tags
                                                                                     ))
    #stmt = stmt.on_conflict_nothing(index_elements=['pageID', 'base_dataID']) #activate this (and take the line above out)
    #                                                                         if "old" rows shall not be updated
    with Session(engine) as session:
        session.execute(stmt)
        session.commit()


def base_to_db(basefile_path = Config.basefile_path):
    df = load_data(basefile_path)
    update_base(df)


