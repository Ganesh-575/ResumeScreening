import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset with encoding fix
df = pd.read_csv("C:\\Users\\halwa\\OneDrive\\Desktop\\R\\R\\UpdatedResumeDataSet.csv", encoding="ISO-8859-1")



# Data Visualization
plt.figure(figsize=(15,5))
sns.countplot(df['Category'])
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(15,10))
counts = df['Category'].value_counts()
labels = df['Category'].unique()
plt.pie(counts, labels=labels, autopct='%1.1f%%', shadow=True, colors=plt.cm.plasma(np.linspace(0,1,len(labels))))
plt.show()

# Oversampling for balanced dataset
max_size = df['Category'].value_counts().max()
df = df.groupby('Category').apply(lambda x: x.sample(max_size, replace=True)).reset_index(drop=True)
df = df.sample(frac=1).reset_index(drop=True)

# Resume Cleaning Function
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)  
    cleanText = re.sub('[%s]' % re.escape("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText) 
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

df['Resume'] = df['Resume'].apply(lambda x: cleanResume(x))

# Encoding Categories
le = LabelEncoder()
df['Category'] = le.fit_transform(df['Category'])

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(df['Resume'])
y = df['Category']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.toarray()
X_test = X_test.toarray()

# Model Training and Evaluation
def train_and_evaluate(model, model_name):
    clf = OneVsRestClassifier(model)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    return clf

knn_model = train_and_evaluate(KNeighborsClassifier(), "KNeighborsClassifier")
svc_model = train_and_evaluate(SVC(), "SVC")
rf_model = train_and_evaluate(RandomForestClassifier(), "RandomForestClassifier")

# Save models
pickle.dump(tfidf, open('tfidf.pkl','wb'))
pickle.dump(svc_model, open('clf.pkl', 'wb'))
pickle.dump(le, open("encoder.pkl",'wb'))

# Resume Prediction Function
def predict_category(input_resume):
    cleaned_text = cleanResume(input_resume) 
    vectorized_text = tfidf.transform([cleaned_text]).toarray()
    predicted_category = svc_model.predict(vectorized_text)
    return le.inverse_transform(predicted_category)[0]

# Streamlit UI
st.title("AI-powered Resume Screening System")
uploaded_file = st.file_uploader("Upload your resume", type=["txt", "pdf", "docx"])
if uploaded_file is not None:
    try:
        resume_text = uploaded_file.read().decode("ISO-8859-1", errors="replace")
    except UnicodeDecodeError:
        resume_text = uploaded_file.read().decode("utf-8", errors="ignore")
    
    category = predict_category(resume_text)
    st.write(f"Predicted Resume Category: {category}")
