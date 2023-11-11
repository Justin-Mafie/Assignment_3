import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('spam_or_not_spam.csv')

# Adding empty columns for new features
data['hyperlink'] = None
data['free'] = None
data['click'] = None
data['business'] = None
data['frequency'] = None

# Define a function to extract the most frequent word ratio
def get_frequency(text):
    if isinstance(text, str):  # Check if 'text' is a string
        words = text.split()
        if len(words) == 0:
            return 0
        word_counts = {word: words.count(word) for word in set(words)}
        most_frequent_word = max(word_counts, key=word_counts.get)
        return word_counts[most_frequent_word] / len(words)
    else:
        return 0

# Iterate through each row and fill in the new columns
for index, row in data.iterrows():
    text = row['email']

    data.at[index, 'hyperlink'] = int('hyperlink' in str(text))
    data.at[index, 'free'] = int('free' in str(text))
    data.at[index, 'click'] = int('click' in str(text))
    data.at[index, 'business'] = int('business' in str(text))
    data.at[index, 'frequency'] = get_frequency(text)

# Extract features and labels as lists
X = data[['hyperlink', 'free', 'click', 'business', 'frequency']].values
y = data['label'].values

# Split the data into training (70%) and testing (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
