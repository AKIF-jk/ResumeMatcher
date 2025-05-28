import os
import PyPDF2
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Extracts text from a PDF file
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# Preprocesses the text by removing special characters, converting to lowercase, and lemmatizing
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def main(resume_folder, job_description_csv, job_limit=None):
    # Result structure
    data = {
        'job_id': [],
        'description': [],
        'resume1': [],
        'resume1_similarity': [],
        'resume2': [],
        'resume2_similarity': [],
        'resume3': [],
        'resume3_similarity': []
    }

    # Loads job descriptions
    jobs_df = pd.read_csv(job_description_csv, usecols=["job_id", "skills_desc", "description"])
    jobs_df["skills_desc"] = jobs_df["skills_desc"].fillna("")
    jobs_df["description"] = jobs_df["description"].fillna("")
    if job_limit is not None:
        jobs_df = jobs_df.head(job_limit)

    # Loads and preprocesses all resumes
    resumes = []
    resume_files = []

    for root, _, files in os.walk(resume_folder):
        for filename in files:
            if filename.lower().endswith('.pdf'):
                path = os.path.join(root, filename)
                text = extract_text_from_pdf(path)
                preprocessed = preprocess_text(text)
                resumes.append(preprocessed)
                # Store the path relative to the data folder
                resume_files.append(os.path.relpath(path, resume_folder))

    # vectorize the resumes
    vectorizer = TfidfVectorizer()
    resume_tfidf = vectorizer.fit_transform(resumes)

    for index, row in jobs_df.iterrows():
        job_id = row['job_id']
        # Concatenates skills_desc and description for matching
        job_text = (row['skills_desc'] + ' ' + row['description']).strip()
        job_preprocessed = preprocess_text(job_text)

        # Computes the cosine similarity between the job description and the resumes
        job_tfidf = vectorizer.transform([job_preprocessed])
        scores = cosine_similarity(job_tfidf, resume_tfidf).flatten()

        # Ranks the top 3 resumes
        top_indices = scores.argsort()[::-1][:3]
        
        data['job_id'].append(job_id)
        data['description'].append(job_text)
        for i in range(3):
            if i < len(top_indices):
                idx = top_indices[i]
                data[f'resume{i+1}'].append(resume_files[idx])
                data[f'resume{i+1}_similarity'].append(round(scores[idx], 4))
            else:
                data[f'resume{i+1}'].append('')
                data[f'resume{i+1}_similarity'].append(0.0)


    result_df = pd.DataFrame(data)
    result_df.to_csv('job_resume_matches.csv', index=False)
    print("Saved results to job_resume_matches.csv")


main('D:/Projects/data', 'D:/Projects/UpdatedPostings.csv', job_limit=100)


