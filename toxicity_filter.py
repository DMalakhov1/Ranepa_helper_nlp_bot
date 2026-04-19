import re
from transformers import pipeline as hf_pipeline 

class ToxicityFilter:
    """
    Фильтер токчисночти 💅

    Инициализация:
        filter = ToxicityFilter(stop_words={"хуй", "хуйня", ...})

    Использование:
        if filter.is_toxic("какой-то текст"):
            print("Токсично!")
    """

    def __init__(self, stop_words: set):
        self.stop_words = stop_words

        # Для фильтрации особо умных бл*дей
        self._substitutions = {                              
            "0": "о", "3": "з", "@": "а", "$": "с",
            "1": "и", "!": "и", "*": "я",
        }

        # загружаем классификатор Skolkovo
        self.ml_classifier = hf_pipeline(
                "text-classification",
                model="SkolkovoInstitute/russian_toxicity_classifier",
            )

    def _normalize(self, text: str) -> str:
        """
        Нормализуем текст
        """       
        # Приводим к нижнему регистру + удалеям всем лишнии пробелы в начале и в конце текста
        text = text.lower().strip()
        
        # Реализуем замену по славярю _substitutions
        for old, new in self._substitutions.items():
            text = text.replace(old, new)

        # Удаляем повторы: "бляяяяяя" → "бля"
        text = re.sub(r"(.)\1{2,}", r"\1", text)
        return text

    def is_toxic(self, text: str) -> bool:
        """
            True если текст токсичный, False если милашка
        """

        # нормализуем
        normalized = self._normalize(text)

        # Извлекаем все слова (только буквы, без цифр и символов)
        words_in_text = set(re.findall(r"[а-яёa-z]+", normalized))

        # Проверяем каждое слово: есть ли оно в словаре?
        for word in words_in_text:
            if word in self.stop_words:
                return True

        # ML-ка Сколково
        # Запускается только если стоп-слова не сработали
        if self.ml_classifier is not None:
            result = self.ml_classifier(text)[0]
            # Результат выглядит так: {'label': 'toxic', 'score': 0.98}
            # Порог 0.70: если модель уверена на 75%+ что токсично - бан
            if result["label"] == "toxic" and result["score"] > 0.7:
                return True
        return False
