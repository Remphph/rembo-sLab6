import requests
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Конфигурация Streamlit
st.title("Анализ фейковых новостей")
st.sidebar.header("Параметры")
country = st.sidebar.selectbox("Выберите страну", options=["us", "ru", "gb", "fr", "de"], index=0)
page_size = st.sidebar.slider("Количество статей", min_value=10, max_value=100, value=10, step=10)

# Параметры API
API_KEY = "a21fac50702949948d8ea5176f7ace49"
URL = "https://newsapi.org/v2/top-headlines"
params = {
    "apiKey": API_KEY,
    "country": country,
    "pageSize": page_size,
}

# Загрузка данных с News API
st.write("Загрузка данных...")
response = requests.get(URL, params=params)
if response.status_code == 200:
    data = response.json()
    articles = data.get("articles", [])
    st.write(f"Получено {len(articles)} статей")
else:
    st.error("Ошибка при загрузке данных с News API")
    articles = []

# Преобразование в DataFrame
if articles:
    news_data = pd.DataFrame(articles)
    st.write("Загруженные данные:")
    st.dataframe(news_data[["source", "title", "description", "content"]])
else:
    st.error("Нет доступных данных для отображения.")
    st.stop()

# Предобработка текста
st.write("Предобработка текста...")
news_data["text"] = news_data["title"] + " " + news_data["description"]
news_data["text"] = news_data["text"].fillna("").str.lower()
news_data["text"] = news_data["text"].str.replace(r'[^\w\s]', '', regex=True)

# Создание целевой переменной (рандомная генерация меток для тестирования)
st.write("Создание целевой переменной...")
np.random.seed(42)
news_data["label"] = np.random.choice([0, 1], size=len(news_data))

# Распределение классов
st.write("Распределение классов:")
st.bar_chart(news_data["label"].value_counts())

# Векторизация текста
st.write("Векторизация текста...")
vectorizer = CountVectorizer(stop_words="english", max_features=1000)
X = vectorizer.fit_transform(news_data["text"])
y = news_data["label"]

# Разделение данных на обучающие и тестовые
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
st.write("Обучение модели Naive Bayes...")
model = MultinomialNB()
model.fit(X_train, y_train)

# Оценка модели
st.write("Оценка модели...")
y_pred = model.predict(X_test)
st.write("Отчет классификации:")
st.text(classification_report(y_test, y_pred))

# Матрица ошибок
st.write("Матрица ошибок:")
cm = confusion_matrix(y_test, y_pred)
st.write(pd.DataFrame(cm, index=["True:0", "True:1"], columns=["Pred:0", "Pred:1"]))

# Визуализация важных слов
st.write("Визуализация важных слов...")
fake_words = " ".join(news_data[news_data["label"] == 1]["text"])
real_words = " ".join(news_data[news_data["label"] == 0]["text"])

col1, col2 = st.columns(2)
with col1:
    st.subheader("Фейковые новости")
    wordcloud_fake = WordCloud(background_color="white").generate(fake_words)
    plt.imshow(wordcloud_fake, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

with col2:
    st.subheader("Правдивые новости")
    wordcloud_real = WordCloud(background_color="white").generate(real_words)
    plt.imshow(wordcloud_real, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)
