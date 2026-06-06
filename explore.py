import pandas as pd

# Load datasets
fake = pd.read_csv('Fake.csv')
real = pd.read_csv('True.csv')

# Add labels
fake['label'] = 0  # 0 = Fake
real['label'] = 1  # 1 = Real

# Combine both
df = pd.concat([fake, real], ignore_index=True)

# Explore
print("Total articles:", len(df))
print("\nColumns:", df.columns.tolist())
print("\nLabel distribution:\n", df['label'].value_counts())
print("\nSample article:\n", df['text'][0][:300])