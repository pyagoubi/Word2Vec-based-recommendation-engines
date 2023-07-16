from setup.Setup import setup
from StoreSourceData.base_to_db import base_to_db
from StoreSourceData.session_to_db import session_to_db
from predict.base_pred import base_predict
from predict.session_pred import session_predict
from config import Config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    """
    Arguments
    setup: only set to True for the very first run (initializes database and downloads base model)
    basefile: Path to source data for the base model (currently set to 'kow.xlsx' in the Config class)
    sessionfile: Path to source data for the session model (currently set to 'kow.xlsx' in the Config class)
    dbconnection: database connection link
    sessionmodel: pathe to were the sessionmodel is to be stored
    recno: number of recommendations per pageID that should be generated
    setup: only set to True for the very first run (initializes database and downloads base model)
    loadbase: whether to load base data into the database source table
    predictbase: whether to generate predictions and store them in the predictions table
    loadsession: whether to load session data into the database source table
    predictsession: whether to generate session based predictions and store them in the predictions table
   
    """



    parser.add_argument("-su","--setup", default=True,          #set to False after first run
                        action=argparse.BooleanOptionalAction)
    parser.add_argument("-bf", "--basefile", default=Config.basefile_path)
    parser.add_argument("-sf", "--sessionfile", default=Config.sessionfile_path)
    parser.add_argument("-db", "--dbconnection", default=Config.dbconnection)
    parser.add_argument("-sm", "--sessionmodel", default=Config.session_model_path)
    parser.add_argument("--recno", type=int, default=Config.rec_no)

    parser.add_argument("-lb","--loadbase", default=True,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument("-pb","--predictbase", default=True,
                         action=argparse.BooleanOptionalAction)
    parser.add_argument("-ls","--loadsession", default=True,
                         action=argparse.BooleanOptionalAction)
    parser.add_argument("-ps","--predictsession", default=True,
                         action=argparse.BooleanOptionalAction)

    args, unknown = parser.parse_known_args()

    if args.setup:
        setup()
    if args.loadbase:
        base_to_db(basefile_path=args.basefile)
    if args.loadsession:
        session_to_db(sessionfile_path=args.sessionfile)
    if args.predictbase:
        base_predict(model_name=Config.model_name,dbconnection=args.dbconnection, recnumber=args.recno)
    if args.predictsession:
        session_predict(session_model_path=args.sessionmodel, dbconnection = args.dbconnection,
                        recnumber = args.recno)


