import pickle
import psycopg2
import pandas as pd
import sklearn
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
nltk.download('stopwords')

def request_data(database_credentials):
    try:
        conn = psycopg2.connect(
            host = database_credentials["server"],
            port = database_credentials["port"],
            dbname = database_credentials["database"],
            user = database_credentials["username"],
            password = database_credentials["password"]
        )
        sql_query = "select publish_datetime, article_title, article_summary, tags, sentiment_score from afr_articles;"
        df = pd.read_sql_query(sql_query, conn)
        conn.close()
        print("Connected and closed")

        return df
    except:
        conn.close()
        print("I am unable to connect to the database")
        return None

def date_average_pairs(df):
    date_sentiment = {}
    unique_dates = list(set(df['date']))
    for date in unique_dates:
        average_sentiment = df[df['date'] == date]['sentiment_score'].mean()
        date_sentiment[date] = average_sentiment
    lists = sorted(date_sentiment.items()) # sorted by key, return a list of tuples
    dates, sentiment = zip(*lists) # unpack a list of pairs into two tuples
    return dates, sentiment

def build_inputs(df, dates):
    inputs = []
    for date in dates:
        titles = df[df['date'] == date]['article_title']
        summaries = df[df['date'] == date]['article_summary']
        tags = df[df['date'] == date]['tags']

        date_string = ""
        for x, y, z in zip(titles, summaries, tags):
            date_string += x + " " + y + " " + z
        inputs.append(date_string)
    return inputs[:-1]

def build_labels(sentiment):
    return sentiment[1:]

def split_data(X, y, test_size):
    return sklearn.model_selection.train_test_split(
        X, y, test_size=test_size)

def train_model(X_train, y_train):
    count_vectorizer = TfidfVectorizer(
        analyzer="word", tokenizer=nltk.word_tokenize,
        preprocessor=None, stop_words='english', max_features=None)

    model = make_pipeline(count_vectorizer, LinearRegression())
    model.fit(X_train, y_train)
    return model


def main(database_credentials):
    # Pull data
    df = request_data(database_credentials)

    # Add date col, drop time col
    df['date'] = list(x.date() for x in df['publish_datetime'])
    df = df.drop(['publish_datetime'], axis=1)

    # Get average sentiment for each date
    dates, sentiment = date_average_pairs(df)

    # Feature engineering
    X = build_inputs(df, dates)
    y = build_labels(sentiment)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y, .02)

    # Training linear regression
    model = train_model(X_train, y_train)

    # Evaluate
    y_predict = model.predict(X_test)
    error = mean_squared_error(y_test, y_predict)
    print("RMSE:", error)

    # Export
    pkl_filename = "model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)


if __name__ == "__main__":
    # CREDS
    database_credentials = {
        "server": None,
        "port": None,
        "database": None,
        "username": None,
        "password": None
    }
    main(database_credentials)
