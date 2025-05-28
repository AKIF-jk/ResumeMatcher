import pandas as pd

# Load the CSV
df = pd.read_csv('D:/Projects/postings.csv/postings.csv')

# Drop rows where the specific column (e.g., 'column_name') is null or empty
df_cleaned = df[df['skills_desc'].notna() & df['skills_desc'].astype(str).str.strip().ne('')]
limited_df = df_cleaned.head(100)
# Save the cleaned data back to the CSV
limited_df.to_csv('UpdatedPostings.csv', index=False)
