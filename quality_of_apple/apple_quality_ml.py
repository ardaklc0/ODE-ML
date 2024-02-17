import pandas as pd

csv_file_path = 'quality_of_apple/apple_quality.csv'
df = pd.read_csv(csv_file_path)
# Omit id column
data = df.drop(['A_id', 'Quality'], axis=1)
expected_output = df['Quality']
# Taking 0.33 of the data for training
train_data = data[:int(len(data) * 0.33)]


print(data)
print(expected_output)
print(train_data)