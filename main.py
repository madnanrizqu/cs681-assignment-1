import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('./imdb_reviews.csv')

train_df, remaining_df = train_test_split(df, train_size=0.6, random_state=42)

val_df, test_df = train_test_split(remaining_df, train_size=0.5, random_state=42)

print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Test set size: {len(test_df)}")