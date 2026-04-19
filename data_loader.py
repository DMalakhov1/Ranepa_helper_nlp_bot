import pandas as pd
import config


def load_FAQ() -> pd.DataFrame:
    """
    Загружает FAQ из Database.xlsx
    Возвращает:
        pd.DataFrame с колонками:
        > Question - вопрос абитуриента
        > Answer - готовый ответ
        > Question type - тип вопроса
    """
    df = pd.read_excel(config.FAQ_PATH)
    # Убираем Nan
    df = df.dropna(subset=["Question", "Answer"])
    # Очищает текст от лишних пробелов
    df["Question"] = df["Question"].astype(str).str.strip()
    df["Answer"] = df["Answer"].astype(str).str.strip()
    return df.reset_index(drop=True)


def load_knowledge_base() -> pd.DataFrame:
    """
    Загружает текстовую базу знаний из Database-2.xlsx
    Возвращает:
        pd.DataFrame с колонками:
        > header - заголовок раздела
        > text - текст раздела
        > chunk - header + "\n" + text
    """
    df = pd.read_excel(config.KB_PATH)
    df = df.dropna(subset=["header", "text"])
    df["header"] = df["header"].astype(str).str.strip()
    df["text"] = df["text"].astype(str).str.strip()

    # Объединяем заголовок и текст в один чанк для RAG
    df["chunk"] = df["header"] + "\n" + df["text"]
    return df.reset_index(drop=True)


def load_programs() -> pd.DataFrame:
    """
    Загружает таблицу образовательных программ из all_program.xlsx
    Возвращает:
        pd.DataFrame с дохуя колонками :)
    """
    df = pd.read_excel(config.PROGRAMS_PATH)

    # Создаём lowercase-версии текстовых колонок для поиска
    # Оставляем изначальные версии для красивой выдачи пользователям
    # К ниждему регистру приводятся только те столбцы, по которым происходит поиск, а не которые дают инфу
    for col in ["program", "megacluster", "institute", "major", "tracks"]:
        if col in df.columns:
            df[col + "_lower"] = df[col].astype(str).str.lower().str.strip()
    return df.reset_index(drop=True)

def load_stop_words() -> set:
    """
    Загружает стоп-слова из двух txt-файлов
    """
    words = set()
    for path in [config.ABUSIVE_WORDS_PATH, config.CURSE_WORDS_PATH]:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                w = line.strip().lower()
                if w:
                    words.add(w)
    return words
