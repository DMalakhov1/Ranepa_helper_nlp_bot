import os
import sqlite3
import re
import pandas as pd
import config


# Путь к файлу БД (рядом с данными)
DB_PATH = os.path.join(config.DATA_DIR, "ranepa.db")


# Создание БД из Excel-файлов 

def rebuild_db(force: bool = False) -> str:
    if os.path.exists(DB_PATH) and not force:
        print(f"  ✅ БД уже существует: {DB_PATH}")
        return DB_PATH

    conn = sqlite3.connect(DB_PATH)

    if os.path.exists(config.FAQ_PATH):
        df_faq = pd.read_excel(config.FAQ_PATH)
        df_faq = df_faq.dropna(subset=["Question", "Answer"])
        df_faq = df_faq.rename(columns={"№": "id"})
        df_faq.to_sql("faq", conn, if_exists="replace", index=False)
        print(f"    faq: {len(df_faq)} строк")

    if os.path.exists(config.KB_PATH):
        df_kb = pd.read_excel(config.KB_PATH)
        df_kb = df_kb.dropna(subset=["header", "text"])
        df_kb.to_sql("knowledge", conn, if_exists="replace", index=False)
        print(f"    knowledge: {len(df_kb)} строк")

    if os.path.exists(config.PROGRAMS_PATH):
        df_prog = pd.read_excel(config.PROGRAMS_PATH)
        for col in df_prog.columns:
            df_prog[col] = df_prog[col].astype(str).replace("nan", "")
        df_prog.to_sql("programs", conn, if_exists="replace", index=False)
        print(f"    programs: {len(df_prog)} строк")

    # Индексы для быстрого поиска
    conn.execute("CREATE INDEX IF NOT EXISTS idx_faq_question ON faq(Question)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_programs_name ON programs(program)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_programs_mega ON programs(megacluster)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_kb_header ON knowledge(header)")
    conn.commit()
    conn.close()

    print(f"✅ БД создана: {DB_PATH}")
    return DB_PATH


def _get_conn() -> sqlite3.Connection:
    """Соединение с Row factory (строки как dict)"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _to_dicts(rows) -> list[dict]:
    """sqlite3.Row → list[dict]."""
    return [dict(r) for r in rows]


def get_program_by_name(name: str) -> dict | None:
    """
    Ищет программу по названию (LIKE '%...%').
    """
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM programs WHERE LOWER(program) LIKE LOWER(?)",
        (f"%{name.strip()}%",)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def search_programs(query: str, limit: int = 10) -> list[dict]:
    """
    Ищет программы по ключевым словам в нескольких колонках
    """
    words = [w for w in re.findall(r"[а-яёa-z]+", query.lower()) if len(w) > 3]
    if not words:
        return []

    score_parts = []
    params = []
    for word in words:
        pattern = f"%{word}%"
        score_parts.append(
            "(CASE WHEN LOWER(program) LIKE ? THEN 5 ELSE 0 END"
            " + CASE WHEN LOWER(edu_form) LIKE ? THEN 10 ELSE 0 END"
            " + CASE WHEN LOWER(megacluster) LIKE ? THEN 2 ELSE 0 END"
            " + CASE WHEN LOWER(major) LIKE ? THEN 2 ELSE 0 END"
            " + CASE WHEN LOWER(tracks) LIKE ? THEN 1 ELSE 0 END"
            " + CASE WHEN LOWER(desc) LIKE ? THEN 1 ELSE 0 END"
            " + CASE WHEN LOWER(skills) LIKE ? THEN 1 ELSE 0 END)"
        )
        params.extend([pattern] * 7)

    score_expr = " + ".join(score_parts)
    sql = f"""
        SELECT *, ({score_expr}) AS _score
        FROM programs
        WHERE ({score_expr}) > 0
        ORDER BY _score DESC
        LIMIT ?
    """
    params = params + params + [limit]

    conn = _get_conn()
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return _to_dicts(rows)


def get_cost(name: str) -> str | None:
    """Стоимость обучения"""
    row = get_program_by_name(name)
    return row["cost"] if row and row.get("cost") else None


def get_ege(name: str, mode: str = "budget") -> str | None:
    """Требования ЕГЭ для бюджетных или платных мест"""
    row = get_program_by_name(name)
    if not row:
        return None
    col = "eges_budget" if mode == "budget" else "eges_contract"
    return row.get(col) or None


def get_pass_score(name: str) -> str | None:
    """Проходной балл 2024"""
    row = get_program_by_name(name)
    return row["pass_2024"] if row and row.get("pass_2024") else None


def get_places(name: str) -> dict | None:
    """Бюджетные + платные места"""
    row = get_program_by_name(name)
    if not row:
        return None
    return {"budget_2025": row.get("budget_2025", "?"),
            "contract_2025": row.get("contract_2025", "?")}


def get_programs_by_megacluster(megacluster: str) -> list[dict]:
    """Все программы мегакластера"""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM programs WHERE LOWER(megacluster) LIKE LOWER(?)",
        (f"%{megacluster.strip()}%",)
    ).fetchall()
    conn.close()
    return _to_dicts(rows)


def compare_programs(name1: str, name2: str) -> list[dict]:
    """Данные двух программ для сравнения."""
    results = []
    for name in [name1, name2]:
        row = get_program_by_name(name)
        if row:
            results.append(row)
    return results


def get_all_faq() -> list[dict]:
    """Все FAQ (для построения индекса эмбеддингов)"""
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM faq").fetchall()
    conn.close()
    return _to_dicts(rows)


def get_all_knowledge() -> list[dict]:
    """Все документы базы знаний"""
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM knowledge").fetchall()
    conn.close()
    return _to_dicts(rows)


def get_all_programs() -> list[dict]:
    """Все программы"""
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM programs").fetchall()
    conn.close()
    return _to_dicts(rows)


# для LLM-генерации SQL

def run_safe_select(sql: str, params: tuple = ()) -> list[dict]:
    """
    Выполняет произвольный SELECT-запрос
    """
    cleaned = sql.strip().upper()
    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE"]
    for word in forbidden:
        if word in cleaned:
            raise ValueError(f"Запрещённая операция: {word}")

    if not cleaned.startswith("SELECT"):
        raise ValueError("Разрешены только SELECT-запросы.")

    conn = _get_conn()
    try:
        rows = conn.execute(sql, params).fetchall()
        return _to_dicts(rows)
    except sqlite3.Error as e:
        print(f"  [SQL ERROR] {e}")
        return []
    finally:
        conn.close()


def get_table_schema() -> str:
    """
    Описание схемы БД для системного промпта LLM.
    Когда LLM генерирует SQL - промпт включает эту схему.
    """
    return """
Таблица `programs` (105 строк) - образовательные программы РАНХиГС:
  program              TEXT  - название программы
  megacluster          TEXT  - мегакластер
  institute            TEXT  - институт
  major                TEXT  - направление подготовки
  tracks               TEXT  - треки/специализации
  qual                 TEXT  - квалификация (бакалавриат / магистратура)
  edu_form             TEXT  - форма обучения
  edu_years            TEXT  - срок обучения (лет)
  pass_2024            TEXT  - проходной балл 2024
  budget_2025          TEXT  - бюджетные места 2025
  contract_2025        TEXT  - платные места 2025
  cost                 TEXT  - стоимость (руб./год)
  desc                 TEXT  - описание программы
  skills               TEXT  - навыки выпускника
  eges_contract        TEXT  - ЕГЭ (контракт)
  eges_budget          TEXT  - ЕГЭ (бюджет)

Таблица `faq` (736 строк):
  Question TEXT, Answer TEXT, "Question type" TEXT

Таблица `knowledge` (123 строки):
  header TEXT, text TEXT

Важно: все значения TEXT. Для числовых сравнений: CAST(cost AS INTEGER).
Названия в нижнем регистре - используй LOWER().
""".strip()


def format_program(row: dict) -> str:
    """Форматирует данные программы в читаемый текст."""
    LABELS = {
        "program": "Программа", "megacluster": "Мегакластер",
        "institute": "Институт", "major": "Направление",
        "tracks": "Треки", "edu_form": "Форма обучения",
        "edu_years": "Срок обучения (лет)", "pass_2024": "Проходной балл 2024",
        "budget_2025": "Бюджетные места 2025", "contract_2025": "Платные места 2025",
        "cost": "Стоимость (руб./год)", "eges_contract": "ЕГЭ (контракт)",
        "eges_budget": "ЕГЭ (бюджет)",
    }
    lines = []
    for col, label in LABELS.items():
        val = row.get(col, "")
        if val and str(val).strip() and str(val).strip().lower() not in ("nan", ""):
            lines.append(f"{label}: {val}")
    return "\n".join(lines)


def format_results(rows: list[dict], max_rows: int = 5) -> str:
    """Форматирует несколько программ в текст для контекста LLM."""
    if not rows:
        return ""
    parts = []
    for i, row in enumerate(rows[:max_rows], 1):
        parts.append(f"--- Программа {i} ---\n{format_program(row)}")
    return "\n\n".join(parts)
