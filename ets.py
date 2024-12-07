import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import matplotlib.pyplot as plt
import re

# Загрузка необходимых ресурсов NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Пример текста песни
song_text = """
Imagine there's no heaven
It's easy if you try
No hell below us
Above us, only sky
Imagine all the people
Living for today
"""

# Очистка текста от лишних символов (например, пустые строки)
cleaned_text = " ".join([line for line in song_text.split('\n') if line.strip() != ""])

# Токенизация текста с использованием регулярных выражений
tokens = re.findall(r'\b\w+\b', cleaned_text.lower())  # Приводим к нижнему регистру

# Нормализация: лемматизация и удаление стоп-слов
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Лемматизация и фильтрация стоп-слов
normalized_tokens = [
    lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word.isalpha()
]

# Вывод токенов
print("Токены:")
print(tokens)

# Вывод нормализованных токенов (лемматизация и удаление стоп-слов)
print("\nНормализованные токены (лемматизация и удаление стоп-слов):")
print(normalized_tokens)

# Анализ частотности слов
word_counts = Counter(normalized_tokens)

# Вывод частоты встречаемости слов
print("\nЧастотность слов:")
print(word_counts.most_common(10))

# Построение графика частотности
top_words = word_counts.most_common(10)
words, counts = zip(*top_words)

# Построение графика
plt.figure(figsize=(10, 6))
plt.bar(words, counts)
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 10 Most Frequent Words in Song Lyrics')
plt.xticks(rotation=45)
plt.show()
