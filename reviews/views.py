# reviews/views.py
from django.shortcuts import render
import joblib
from nltk.tokenize import word_tokenize
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk
import os
from googletrans import Translator
translator = Translator()
# Загрузка NLTK данных (можно оставить, если они не загружены)
nltk.download('stopwords', quiet=True)  # quiet=True подавляет вывод
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
# Загрузка сохранённых моделей
model_path = os.path.join(os.path.dirname(__file__), 'model_status.pkl')
model = joblib.load(model_path)  # Удалили лишний аргумент
Tfvector = joblib.load(os.path.join(os.path.dirname(__file__), 'tfidf_vectorizer.pkl'))  # Исправлено
label_encoder = joblib.load(os.path.join(os.path.dirname(__file__), 'label_encoder.pkl'))  # Исправлено

# Функция для предобработки текста
def preprocess_text(text: str) -> str:
    cleaned_text = re.sub(r"[^\w\s]", "", text)  # Удаление знаков препинания
    tokens = word_tokenize(cleaned_text)  # Токенизация
    stop_words = set(stopwords.words("english"))  # Список стоп-слов
    tokens_without_stopwords = [token for token in tokens if token.lower() not in stop_words]  # Удаление стоп-слов
    stemmer = SnowballStemmer(language="english")  # Стемминг
    stemmed_tokens = [stemmer.stem(token) for token in tokens_without_stopwords]  # Стемминг
    return ' '.join(stemmed_tokens)  # Возвращение предобработанного текста как строки


# Восстановление TF-IDF Vectorizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# Tfvector = TfidfVectorizer(vocabulary=tfidf_vocabulary, tokenizer=preprocess_text)
# Tfvector.fit([])

def classify_review(text):
    # Предобработка текста
    # preprocessed_text = preprocess_text(text)
    # # Преобразование текста в TF-IDF
    # text_tfidf = Tfvector.transform([text])
    # # Предсказание
    # prediction = model.predict(text_tfidf)
    # # Расшифровка метки
    # label = label_encoder.inverse_transform(prediction)[0]

    return label_encoder.inverse_transform(model.predict(Tfvector.transform([preprocess_text(str(text))])))[0]


def review_input(request):
    if request.method == 'POST':
        review_text = request.POST.get('review')
        original_review_text = review_text
        lang_detected = translator.detect(review_text).lang
        if lang_detected == 'ru':
            review_text = translator.translate(review_text, dest='en').text
       
        predicted_label = classify_review(review_text)

        if predicted_label > 5:
            result = "Positive"
        else:
            result = "Negative"

        # Форматируем predicted_label для отображения
        formatted_label = f"{predicted_label} ({result})"

        context = {
            'review': original_review_text,
            'predicted_label': formatted_label,  # Обновлено для отображения с результатом
        }

        return render(request, 'reviews/result.html', context)
    return render(request, 'reviews/index.html')
# views.py
from django.shortcuts import render

def revievers_view(request):
    return render(request, 'reviews/index.html')  # Измените на нужный вам шаблон

# reviews/views.py
# from django.shortcuts import render
# import joblib
# from nltk.tokenize import word_tokenize
# import re
# from nltk.corpus import stopwords
# from nltk.stem import SnowballStemmer
# import nltk
# import os
# from sklearn.feature_extraction.text import TfidfVectorizer
#
# # Загрузка NLTK данных
# nltk.download('stopwords', quiet=True)
# nltk.download('punkt', quiet=True)
# nltk.download('punkt_tab', quiet=True)
#
# # Загрузка сохранённых моделей
# model_path = os.path.join(os.path.dirname(__file__), 'model_status.pkl')
# model = joblib.load(model_path)  # Удалили лишний аргумент
# tfidf_vocabulary = joblib.load(os.path.join(os.path.dirname(__file__), 'tfidf_vocabulary.pkl'))  # Исправлено
# label_encoder = joblib.load(os.path.join(os.path.dirname(__file__), 'label_encoder.pkl'))  # Исправлено
#
# # Функция для предобработки текста
# def preprocess_text(text: str, language: str = "english") -> list:
#     cleaned_text = re.sub(r"[^\w\s]", "", text)
#     tokens = word_tokenize(cleaned_text, language=language)
#     stop_words = set(stopwords.words(language))
#     tokens_without_stopwords = [token for token in tokens if token.lower() not in stop_words]
#     stemmer = SnowballStemmer(language=language)
#     stemmed_tokens = [stemmer.stem(token) for token in tokens_without_stopwords]
#     return stemmed_tokens
#
# # Восстановление TF-IDF Vectorizer
# Tfvector = TfidfVectorizer(vocabulary=tfidf_vocabulary, tokenizer=preprocess_text)
#
# def classify_review(text):
#     try:
#         preprocessed_text = preprocess_text(text)
#         text_tfidf = Tfvector.transform([' '.join(preprocessed_text)])  # Преобразование в строку
#         prediction = model.predict(text_tfidf)
#         label = label_encoder.inverse_transform(prediction)[0]
#         return label
#     except Exception as e:
#         print(f"Error in classify_review: {e}")
#         return "Unknown"
#
# def review_input(request):
#     if request.method == 'POST':
#         review_text = request.POST.get('review', '')  # Убедитесь, что возвращается пустая строка по умолчанию
#         predicted_label = classify_review(review_text)
#         context = {
#             'review': review_text,
#             'predicted_label': predicted_label
#         }
#         return render(request, 'reviews/result.html', context)
#     return render(request, 'reviews/index.html')
