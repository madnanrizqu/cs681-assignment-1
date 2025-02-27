import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('./imdb_reviews.csv')

train_df, remaining_df = train_test_split(df, train_size=0.6, random_state=42)

val_df, test_df = train_test_split(remaining_df, train_size=0.5, random_state=42)

print("DataFrame columns:", df.columns)
print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Test set size: {len(test_df)}")

# Create bag of words features
vectorizer = CountVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_df['review'])
X_val = vectorizer.transform(val_df['review'])
X_test = vectorizer.transform(test_df['review'])

# # Train Logistic Regression
# lr_model = LogisticRegression(max_iter=1000)
# lr_model.fit(X_train, train_df['sentiment'])
# lr_pred = lr_model.predict(X_test)
# print("\nLogistic Regression Results:")
# print(classification_report(test_df['sentiment'], lr_pred))

# # Train Naive Bayes
# nb_model = MultinomialNB()
# nb_model.fit(X_train, train_df['sentiment'])
# nb_pred = nb_model.predict(X_test)
# print("\nNaive Bayes Results:")
# print(classification_report(test_df['sentiment'], nb_pred))