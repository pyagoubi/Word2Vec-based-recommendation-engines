
from config import Config
import gensim
import gensim.downloader as api
from sqlalchemy import create_engine,  BigInteger, Integer, String
from sqlalchemy.orm import declarative_base, mapped_column
import os


def create_db(dbconnection = Config.dbconnection):
    Base = declarative_base()

    class Base_Predictions(Base):
        __tablename__ = 'base_predictions'

        base_dataID = mapped_column('base_dataID', Integer, primary_key=True)
        pageID = mapped_column('pageID', BigInteger, primary_key=True)
        rec_no = mapped_column('rec_no', Integer, primary_key=True)
        base_rec = mapped_column('base_rec', BigInteger)

        def __init__(self,  base_dataID, pageID, rec_no, base_rec):

            self.base_dataID = base_dataID
            self.pageID = pageID
            self.rec_no = rec_no
            self.base_rec = base_rec


    class Session_Predictions(Base):
        __tablename__ = 'session_predictions'

        pageID = mapped_column('pageID', BigInteger, primary_key=True)
        rec_no = mapped_column('rec_no', Integer, primary_key=True)
        session_rec = mapped_column('session_rec', BigInteger)

        def __init__(self,  pageID, rec_no, session_rec):
            self.pageID = pageID
            self.rec_no = rec_no
            self.session_rec = session_rec

    class session_dataID_model(Base):
        __tablename__ = 'session_dataID_model'

        index = mapped_column('index', Integer, primary_key = True)
        session_dataID = mapped_column('session_dataID', Integer)

        def __init__(self, session_dataID):
            self.index = index
            self.session_dataID = session_dataID

    class base_source(Base):
        __tablename__ = 'base_source'

        base_dataID = mapped_column('base_dataID', Integer, primary_key=True)
        pageID = mapped_column('pageID', BigInteger, primary_key = True)
        page_title = mapped_column('page_title', String)
        tags = mapped_column('tags', String)

        def __init__(self, base_dataID, pageID):
            self.base_dataID = base_dataID
            self.pageID = pageID
            self.page_title = page_title
            self.tags = tags


    class session_source(Base):
        __tablename__ = 'session_source'

        session_dataID = mapped_column('session_dataID', Integer, primary_key=True)
        sessionID = mapped_column('sessionID', BigInteger, primary_key = True)
        order = mapped_column('order', Integer, primary_key = True)
        page = mapped_column('page', BigInteger)
        userID = mapped_column('userID', BigInteger)

        def __init__(self, session_dataID, sessionID, order, page, userID):
            self.session_dataID = session_dataID
            self.sessionID = sessionID
            self.order = order
            self.page = page
            self.userID = userID

    engine = create_engine(dbconnection, echo = False)
    Base.metadata.create_all(bind=engine)


def load_base_model(model_name=Config.model_name, model_name_orig=Config.model_name_orig):
    if not os.path.isfile(model_name):
        model = api.load(model_name_orig)
        model.save(model_name)


def setup():
    create_db()
    load_base_model(model_name=Config.model_name, model_name_orig=Config.model_name_orig)
