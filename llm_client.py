import config
import db_manager

try:
    from openai import OpenAI
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False
    print("⚠️  openai не установлен: pip install openai")


# Промт для генерации SQL
# Включает: схему таблицы + правила + примеры реальных запросов.

SQL_GENERATION_PROMPT = f"""Ты - SQL-эксперт. Генерируй ОДИН SELECT-запрос к SQLite.

{db_manager.get_table_schema()}

ПРАВИЛА:
1. Возвращай ТОЛЬКО SQL. Без пояснений, без markdown, без ```.
2. Названия программ в нижнем регистре. Ищи через LIKE '%слово%'.
3. Для числовых сравнений: CAST(cost AS INTEGER).
4. LIMIT 10 если не указано иное.
5. Если спрашивают про ЕГЭ - выбирай eges_budget и eges_contract.
6. Если спрашивают про одну программу - LIMIT 1.
7. Заканчивай запрос точкой с запятой.
8. Если нужно сравнить программы - используй IN или UNION.
9.Формы обучения в базе: 'Очная', 'Заочная', 'Очно-заочная'. Если пользователь пишет 'вечернее', ищи 'очно-заочная'. Используй LIKE '%очно-заоч%'.

ПРИМЕРЫ:

Вопрос: Сколько стоит бизнес-информатика?
SQL: SELECT program, cost FROM programs WHERE LOWER(program) LIKE '%бизнес-информатик%' LIMIT 1;

Вопрос: Какие ЕГЭ нужны для анализа данных?
SQL: SELECT program, eges_budget, eges_contract FROM programs WHERE LOWER(program) LIKE '%анализ данн%' LIMIT 1;

Вопрос: Проходной балл на юриспруденцию?
SQL: SELECT program, pass_2024 FROM programs WHERE LOWER(program) LIKE '%юриспруденц%';

Вопрос: Бюджетные места на журналистике?
SQL: SELECT program, budget_2025, contract_2025 FROM programs WHERE LOWER(program) LIKE '%журналист%' LIMIT 1;

Вопрос: Какие программы дешевле 400000?
SQL: SELECT program, cost FROM programs WHERE CAST(cost AS INTEGER) < 400000 AND cost != '' ORDER BY CAST(cost AS INTEGER) LIMIT 10;

Вопрос: Сравни бизнес-информатику и анализ данных
SQL: SELECT program, cost, pass_2024, budget_2025, contract_2025, eges_budget FROM programs WHERE LOWER(program) LIKE '%бизнес-информатик%' OR LOWER(program) LIKE '%анализ данн%';

Вопрос: Программы мегакластера информационные технологии
SQL: SELECT program, cost, pass_2024, budget_2025 FROM programs WHERE LOWER(megacluster) LIKE '%информационные технологии%';

Вопрос: Расскажи про программу менеджмент спортивной индустрии
SQL: SELECT program, megacluster, institute, major, tracks, edu_form, edu_years, pass_2024, budget_2025, contract_2025, cost, eges_budget, eges_contract, desc, skills FROM programs WHERE LOWER(program) LIKE '%менеджмент%спорт%' LIMIT 1;

Вопрос: Самая дорогая программа
SQL: SELECT program, cost FROM programs WHERE cost != '' ORDER BY CAST(cost AS INTEGER) DESC LIMIT 5;

Вопрос: Самая дешёвая программа
SQL: SELECT program, cost FROM programs WHERE cost != '' ORDER BY CAST(cost AS INTEGER) ASC LIMIT 5;

Вопрос: Где больше всего бюджетных мест?
SQL: SELECT program, budget_2025 FROM programs WHERE budget_2025 != '' AND budget_2025 != '0' ORDER BY CAST(budget_2025 AS INTEGER) DESC LIMIT 10;
"""


class LLMClient:
    def __init__(self):
        self.client = None
        self.model = config.LLM_MODEL
        if _HAS_OPENAI and config.LLM_API_KEY:
            self.client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
        elif not config.LLM_API_KEY:
            print("⚠️  OPENAI_API_KEY не задан — LLM-ответы будут недоступны")

    def is_available(self) -> bool:
        return self.client is not None

    def generate(self, system_prompt: str, user_message: str, temperature: float = 0.3) -> str:
        if not self.is_available():
            return "[LLM недоступен — задайте OPENAI_API_KEY в .env]"
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                max_tokens=1500,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"Ошибка LLM: {e}"

    # Генерация SQL
    def generate_sql(self, query: str) -> str:
        if not self.is_available():
            return ""

        user_msg = f"Вопрос абитуриента: {query}\nSQL:"
        raw = self.generate(SQL_GENERATION_PROMPT, user_msg, temperature=0.0)
        return self._clean_sql(raw)

    # Красивое оформление ответа из SQL-данных
    def format_sql_answer(self, query: str, rows: list[dict], sql: str) -> str:
        if not self.is_available():
            # Без LLM — форматируем вручную
            return self._fallback_format(rows)

        # Превращаем строки SQL-результата в текст для LLM
        data_text = self._format_rows(rows)

        user_msg = (
            f"Вопрос абитуриента: {query}\n\n"
            f"Данные из базы (результат SQL-запроса):\n{data_text}\n\n"
            f"Сформулируй точный, понятный ответ на основе этих данных.\n"
            f"Используй *жирный* для ключевых цифр. Не выдумывай данные."
        )
        return self.generate(config.SYSTEM_PROMPT_SQL, user_msg)

    # Классификатор
    def classify_query(self, query: str, prog_ctx: str, kb_ctx: str, faq_ctx: str) -> str:
        if not self.is_available():
            return "general"

        p_preview = prog_ctx[:2000] if prog_ctx else "Нет данных"
        f_preview = faq_ctx[:2000] if faq_ctx else "Нет данных"
        k_preview = kb_ctx[:2000] if kb_ctx else "Нет данных"

        prompt = (
            f"Ты - интеллектуальный маршрутизатор приемной комиссии.\n"
            f"Определи тип вопроса. Ответь ОДНИМ словом: 'program_data' или 'general'.\n\n"
            f"Вопрос: «{query}»\n\n"
            f"Фрагменты из баз:\n"
            f"1. Таблица ОП: {p_preview}\n"
            f"2. FAQ: {f_preview}\n"
            f"3. База Знаний: {k_preview}\n\n"
            f"'program_data' - вопрос о конкретной программе, стоимости, баллах, ЕГЭ, местах, "
            f"сравнение программ, рейтинг по цене/баллам.\n"
            f"'general' - общие вопросы (мегакластеры, правила, документы), "
            f"И если ответ уже есть в FAQ или База Знаний.\n\n"
            f"Тип:"
        )
        try:
            result = self.generate(config.SYSTEM_PROMPT_CLASSIFY, prompt, temperature=0.0)
            return "program_data" if "program" in result.lower() else "general"
        except Exception:
            return "general"

    @staticmethod
    def _clean_sql(raw: str) -> str:
        """Извлекает чистый SQL из ответа LLM."""
        sql = raw.replace("```sql", "").replace("```", "").strip()
        lines = sql.split("\n")
        sql_lines = []
        started = False
        for line in lines:
            stripped = line.strip()
            if stripped.upper().startswith("SELECT"):
                started = True
            if started:
                sql_lines.append(line)
                if stripped.endswith(";"):
                    break
        sql = "\n".join(sql_lines).strip()
        if sql and not sql.endswith(";"):
            sql += ";"
        return sql

    @staticmethod
    def _format_rows(rows: list[dict], max_rows: int = 10) -> str:
        """Форматирует SQL-результат в текст для LLM."""
        if not rows:
            return "(пусто)"
        parts = []
        for i, row in enumerate(rows[:max_rows], 1):
            fields = []
            for k, v in row.items():
                if k.startswith("_"):  # пропускаем служебные (_score)
                    continue
                if v and str(v).strip() and str(v).strip().lower() not in ("nan", ""):
                    fields.append(f"  {k}: {v}")
            if fields:
                parts.append(f"[{i}]\n" + "\n".join(fields))
        return "\n\n".join(parts)

    @staticmethod
    def _fallback_format(rows: list[dict]) -> str:
        """Форматирование без LLM - просто красиво выводим данные."""
        if not rows:
            return "Информация не найдена."
        parts = ["📊 Вот что я нашёл:\n"]
        for row in rows[:5]:
            name = row.get("program", "?")
            parts.append(f"*{name}*")
            for key in ["cost", "pass_2024", "budget_2025", "contract_2025",
                         "eges_budget", "eges_contract", "edu_form", "edu_years"]:
                val = row.get(key, "")
                if val and str(val).strip() and str(val).lower() not in ("nan", ""):
                    label = {
                        "cost": "Стоимость", "pass_2024": "Проходной балл 2024",
                        "budget_2025": "Бюджет мест", "contract_2025": "Платных мест",
                        "eges_budget": "ЕГЭ (бюджет)", "eges_contract": "ЕГЭ (контракт)",
                        "edu_form": "Форма", "edu_years": "Срок",
                    }.get(key, key)
                    parts.append(f"  {label}: {val}")
            parts.append("")
        return "\n".join(parts)
