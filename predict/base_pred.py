import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import numpy as np
import gensim
from scipy.spatial import distance_matrix
import sqlalchemy
from sqlalchemy import create_engine, MetaData, select
from sqlalchemy.orm import Session
from config import Config
from sqlalchemy.dialects.sqlite import insert as sqlite_upsert
nltk.download('stopwords')


def load_preprocess():
    engine = create_engine(Config.dbconnection)

    metadata = MetaData()
    metadata.reflect(bind=engine)
    base_source = sqlalchemy.Table('base_source', metadata, autoload=True)

    stmt = select(base_source).where(base_source.c.base_dataID == select(sqlalchemy.func.max(base_source.c.base_dataID)))
    df = pd.read_sql(stmt, engine)  # load sql table into pandas dataframe

    cols = ['page_title', 'tags', 'tags']  # Columns for similarity measure, 'Tags' is weighed double
    #
    df['combined'] = df[cols].apply(lambda row: ','.join(row.values.astype(str)), axis=1)

    return df

def wordlist(text, remove_stopwords=True):

    """Function to convert a document to a sequence of words,
    optionally removing stop words.  Returns a list of words."""


    # Remove non-letters
    text = re.sub("[^a-zA-Z]"," ", text)
    #
    # Convert words to lower case and split them
    words = text.lower().split()
    #
    # Optionally remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)

def create_clean_list(word_collection_dirty):
    clean_list=[]
    for text in word_collection_dirty: #data_pages['combined']:
      clean_list.append(wordlist(text))
    return clean_list


def makeFeatureVec(words, model, num_features):
    '''Function to average all of the word
    vectors in a given field'''

    # Pre-initialize an empty numpy array
    featureVec = np.zeros((50,),dtype="float32")
    #
    nwords = 0.
    #
    # Index_to_key is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set
    index2word_set = set(model.index_to_key)
    #
    # Loop over each word and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    #
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(word_collection, model, num_features=Config.num_features):
    """calculate the average feature vector for each one and return a 2D numpy array"""

    # Initialize a counter
    counter = 0

    # Preallocate a 2D numpy array
    reviewFeatureVecs = np.zeros((len(word_collection),num_features),dtype="float32")

    # Loop through the word_collection
    for word_vecs in word_collection:

       # Call the function makeFeatureVec that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(word_vecs, model, num_features)
       #
       # Increment the counter
       counter += 1
    return reviewFeatureVecs


def create_predictions(df, datavecs, recnumber):
    """Function for (pre-)predictions that returns a prediction dataframe"""

    dist_cols = [f'pred_{i}' for i in df['pageID']]

    #calculate distance between all vectors
    dist_matrix = pd.DataFrame(distance_matrix(datavecs, datavecs, p=2), columns = dist_cols)
    pred_df_ = pd.concat([df['pageID'].reset_index(drop=True), dist_matrix], axis = 1)
    prediction_df = pd.DataFrame(columns = ['base_dataID', 'pageID', 'rec_no', 'base_rec'])

    did = df['base_dataID'].max()
    datalist= []
    pagelist = []
    recno =[]
    baserec =[]

    for page in df['pageID']:
        baserec.extend(list(pred_df_.sort_values(by=[f'pred_{page}'])['pageID'])[1:recnumber+1])
        datalist.extend((recnumber)*[did])
        pagelist.extend((recnumber)*[page])
        recno.extend(range(1,recnumber+1))

    prediction_df['base_dataID'] = datalist
    prediction_df['pageID'] = pagelist
    prediction_df['rec_no'] = recno
    prediction_df['base_rec'] = baserec

    return prediction_df


def pred_to_db(df, dbconnection=Config.dbconnection):
    """This function updates te predictions in the prediction database"""

    engine = create_engine(dbconnection)

    metadata = MetaData()
    metadata.reflect(bind=engine)
    base_predictions = sqlalchemy.Table('base_predictions', metadata, autoload=True)

    listToWrite = df.to_dict(orient='records')
    stmt = sqlite_upsert(base_predictions).values(listToWrite)
    stmt = stmt.on_conflict_do_update(index_elements=['pageID', 'base_dataID', 'rec_no'], set_=dict(base_rec=stmt.excluded.base_rec
                                                                                     ))
    #stmt = stmt.on_conflict_nothing(index_elements=['pageID', 'base_dataID']) #activate this (and take the line above out)
    #                                                                         if "old" rows shall not be updated
    with Session(engine) as session:
        session.execute(stmt)
        session.commit()


def base_predict(model_name=Config.model_name,dbconnection=Config.dbconnection, recnumber=Config.rec_no):
    model = gensim.models.KeyedVectors.load(model_name)
    df = load_preprocess()
    datavecs = getAvgFeatureVecs(create_clean_list(df['combined']), model, Config.num_features )
    df = create_predictions(df, datavecs, recnumber)
    pred_to_db(df, dbconnection)


