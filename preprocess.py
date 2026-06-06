import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# Load datasets
fake = pd.read_csv('Fake.csv')
real = pd.read_csv('True.csv')

# Add labels
fake['label'] = 0
real['label'] = 1

# Combine
df = pd.concat([fake, real], ignore_index=True)

# Combine title + text into one column
df['content'] = df['title'] + " " + df['text']

# Clean text function
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove stopwords
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# Apply cleaning
df['cleaned'] = df['content'].apply(clean_text)

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save cleaned data
df[['cleaned', 'label']].to_csv('cleaned_data.csv', index=False)

print("✅ Preprocessing done!")
print("Sample cleaned article:\n", df['cleaned'][0][:300])