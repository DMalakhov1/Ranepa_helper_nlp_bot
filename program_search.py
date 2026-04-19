import pandas as pd
import re
from thefuzz import process as fuzz_process, fuzz


SYNONYMS = {
    "it": "информационные технологии", "ит": "информационные технологии",
    "айти": "информационные технологии", "информатик": "информационные технологии",
    "ии": "искусственный интеллект", "ai": "искусственный интеллект",
    "нейросет": "искусственный интеллект", "ml": "машинное обучение",
    "финанс": "финансы экономика банк", "банк": "финансы экономика банк деньги",
    "бухгалтер": "экономика учёт финансы", "аудит": "экономика учёт финансы",
    "юрист": "юриспруденция право", "адвокат": "юриспруденция право",
    "закон": "юриспруденция право", "суд": "юриспруденция право",
    "менеджер": "менеджмент управление", "бизнес": "менеджмент бизнес управление",
    "предприниматель": "менеджмент бизнес", "стартап": "менеджмент бизнес",
    "журналист": "журналистика медиа", "сми": "журналистика медиа",
    "медиа": "журналистика медиа", "блогер": "журналистика медиа",
    "чиновник": "государственное управление", "госслужб": "государственное управление",
    "политик": "политика государственное управление",
    "маркетолог": "маркетинг реклама", "реклам": "маркетинг реклама",
    "pr": "маркетинг реклама", "пиар": "маркетинг реклама", "smm": "маркетинг реклама",
    "туризм": "туризм гостеприимство", "спорт": "спорт спортивная индустрия",
    "аналитик": "анализ данных аналитика", "data": "анализ данных",
    "программист": "разработка программирование информатика",
    "разработчик": "разработка программирование",
    
    "игсу": "институт государственной службы и управления",
    "иэмит": "институт экономики математики и информационных технологий",
    "ион": "институт общественных наук",
    "фэсн": "факультет экономических и социальных наук",
    "ипнб": "институт права и национальной безопасности",
    "иша": "институт широкого профиля",
    "вшфим": "высшая школа финансов и менеджмента",
    "ибда": "институт бизнеса и делового администрирования",
    "иамм": "институт отраслевого менеджмента", # Часто называют ИОМ
    "иом": "институт отраслевого менеджмента",
    "ффиб": "факультет финансов и банковского дела",
    "вшгу": "высшая школа государственного управления",
    "ифиж": "институт филологии и истории",
    "мигсу": "международный институт государственной службы и управления",
    "ифи": "институт филологии и истории",
}


def expand_query(query: str) -> str:
    """
    Расширяет запрос синонимами
    """
    q = query.lower()
    expanded = q
    q_words = set(re.findall(r"[а-яёa-z]+", q))
    for short, full in SYNONYMS.items():
        if len(short) <= 3:
            if short in q_words:
                expanded += " " + full
        else:
            if short in q:
                expanded += " " + full
    return expanded


class ProgramSearchEngine:
    """Поиско по таблице образовательных программ"""

    DISPLAY_COLS = {
        "program":       "📌 Программа",
        "megacluster":   "🏛 Мегакластер",
        "institute":     "🏫 Институт",
        "major":         "📐 Направление подготовки",
        "tracks":        "🛤 Треки (специализации)",
        "qual":          "🎓 Квалификация",
        "edu_form":      "📅 Форма обучения",
        "edu_years":     "⏳ Срок обучения",
        "pass_2024":     "📊 Проходной балл (2024)",
        "budget_2025":   "🆓 Бюджетных мест (2025)",
        "contract_2025": "💳 Платных мест (2025)",
        "cost":          "💰 Стоимость в год (руб.)",
        "eges_budget":   "📝 ЕГЭ на бюджет",
        "eges_contract": "📝 ЕГЭ на контракт",
    }

    SUBJECT_ALIASES = {
        # русский
        "русский":        "русский язык",
        "русский язык":   "русский язык",
        "рус":            "русский язык",
        "рус яз":         "русский язык",
        "русяз":          "русский язык",
 
        # математика
        "математика":     "математика",
        "матем":          "математика",
        "мат":            "математика",
        "матан":          "математика",
        "профиль":        "математика",        
        "профильная":     "математика",
        "профильная математика": "математика",
 
        # информатика
        "информатика":    "информатика",
        "информ":         "информатика",
        "инф":            "информатика",
        "инфа":           "информатика",
        "икт":            "информатика",
 
        # обществознание
        "обществознание": "обществознание",
        "общество":       "обществознание",
        "общ":            "обществознание",
        "общага":         "обществознание",  
        "общество":       "обществознание",
 
        # история
        "история":        "история",
        "ист":            "история",
        "истор":          "история",
 
        # физика
        "физика":         "физика",
        "физ":            "физика",
        "физа":           "физика",
 
        # иностранный язык
        "английский":     "иностранный язык",
        "английский язык": "иностранный язык",
        "иностранный":    "иностранный язык",
        "иностранный язык": "иностранный язык",
        "англ":           "иностранный язык",
        "инглиш":         "иностранный язык",
        "english":        "иностранный язык",
        "немецкий":       "иностранный язык",
        "французский":    "иностранный язык",
 
        # литература
        "литература":     "литература",
        "лит":            "литература",
        "лит-ра":         "литература",
        "литра":          "литература",
 
        # география
        "география":      "география",
        "гео":            "география",
        "геогр":          "география",
 
        # биология
        "биология":       "биология",
        "био":            "биология",
        "биол":           "биология",
 
        # химия
        "химия":          "химия",
        "хим":            "химия",
        "химоза":         "химия",
    }

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        #Список названий
        self.program_names_lower = [str(p).lower() for p in self.df["program"].tolist()]
        self.df["_search_text"] = self.df.apply(self._build_search_text, axis=1)

    @staticmethod
    def _build_search_text(row) -> str:
        parts = []
        for col in ["program", "megacluster", "major", "tracks", "desc", "skills",
                     "megacluster_desc", "institute"]:
            val = str(row.get(col, ""))
            if val and val.lower() != "nan":
                parts.append(val.lower())
        return " ".join(parts)

    # Поиск программ
    def find_programs(self, query: str) -> pd.DataFrame:
        """Четырёхуровневый поиск программы по свободному тексту"""
        q = query.lower().strip()

        # 1. Точное вхождение в название, если чел ввёл полное название программы
        mask = self.df["program_lower"].str.contains(q, na=False, regex=False)
        if mask.any():
            return self.df[mask].copy()

        # 2. По словам запроса, если пользователь сократил название
        q_words = [w for w in re.findall(r"[а-яёa-z]+", q) if len(w) > 3]
        if q_words:
            name_scores = []
            for name in self.program_names_lower:
                score = sum(1 for w in q_words if w[:4] in name)
                name_scores.append(score)
            best = max(name_scores) if name_scores else 0
            if best >= 2:
                result = self.df.copy()
                result["_ns"] = name_scores
                result = result[result["_ns"] >= best].drop(columns=["_ns"])
                return result.head(5)

        # 3. Нечёткий поиск, те если чел написал программу, но криво
        best = fuzz_process.extractOne(q, self.program_names_lower, scorer=fuzz.partial_ratio)
        if best and best[1] >= 75:
            result = self.df[self.df["program_lower"] == best[0]]
            if not result.empty:
                return result.copy()

        # 4. Синонимы + полнотекстовый поиск по описанию/навыкам, если чел вообще не написал программу
        expanded = expand_query(q) # добавляем синонимы
        words = [w for w in re.findall(r"[а-яёa-z]+", expanded) if len(w) > 3]
        if not words:
            return pd.DataFrame()

        scores = []
        for _, row in self.df.iterrows():
            name_text = " ".join([
                str(row.get("program_lower", "")), str(row.get("megacluster_lower", "")),
                str(row.get("major_lower", "")), str(row.get("tracks_lower", "")),
            ])
            desc_text = row.get("_search_text", "")
            name_hits = sum(3 for w in words if w in name_text)
            desc_hits = sum(1 for w in words if w in desc_text and w not in name_text)
            scores.append(name_hits + desc_hits)

        result = self.df.copy()
        result["_score"] = scores
        result = result[result["_score"] >= 3].sort_values("_score", ascending=False)
        return result.drop(columns=["_score"]).head(10)

    # Прямой ответ без LLM, используя find_programs даём ответ
    def try_direct_answer(self, query: str) -> str | None:
        """Прямой ответ (стоимость, ЕГЭ, баллы, места) без вызова LLM"""
        results = self.find_programs(query)
        if results.empty:
            return None

        row = results.iloc[0]
        q = query.lower()
        name = str(row["program"]).strip()

        if any(w in q for w in ["стоимость", "стоит", "цена", "прайс", "руб", "оплат", "денег"]):
            return f"💰 Стоимость обучения на программе «{name}»: *{row.get('cost', '?')} руб./год*"
        if any(w in q for w in ["егэ", "предмет", "экзамен"]):
            return (f"📝 Требования к ЕГЭ (программа «{name}»):\n\n"
                    f"*Бюджет:* {row.get('eges_budget', '?')}\n*Контракт:* {row.get('eges_contract', '?')}")
        if any(w in q for w in ["бюджет", "мест", "квот"]):
            return (f"🎓 Места на программе «{name}»:\n"
                    f"*Бюджет:* {row.get('budget_2025', '?')}\n*Контракт:* {row.get('contract_2025', '?')}")
        if any(w in q for w in ["балл", "проходн"]):
            return f"📈 Проходной балл на программу «{name}» (2024): *{row.get('pass_2024', '?')}*"
        return None

    
    # Обработка кнопок в тг по пронграмме 
    def get_program_field(self, program_name: str, field: str) -> str | None:
        """Возвращает конкретное поле программы (для inline-кнопок)."""
        mask = self.df["program_lower"].str.contains(program_name.lower(), na=False, regex=False)
        if not mask.any():
            return None
        row = self.df[mask].iloc[0]
        name = str(row["program"]).strip()
        fields = {
            "cost": f"💰 Стоимость «{name}»: *{row.get('cost', '?')} руб./год*",
            "ege": (f"📝 ЕГЭ «{name}»:\n\n*Бюджет:* {row.get('eges_budget', '?')}\n"
                    f"*Контракт:* {row.get('eges_contract', '?')}"),
            "score": f"📈 Проходной балл «{name}» (2024): *{row.get('pass_2024', '?')}*",
            "places": (f"🎓 Места «{name}»:\n*Бюджет:* {row.get('budget_2025', '?')}\n"
                       f"*Контракт:* {row.get('contract_2025', '?')}"),
            "full": self.format_program_info(row),
        }
        return fields.get(field)

    # Подбор по ЕГЭ
    @staticmethod
    def _parse_ege_requirements(ege_text: str) -> dict:
        if not isinstance(ege_text, str) or not ege_text.strip():
            return {}
        result = {}
        for match in re.finditer(r'([а-яё\s\-]+?):\s*([\d.]+)', ege_text.lower()):
            subject = match.group(1).strip()
            score = int(float(match.group(2)))
            subject = re.sub(r'^.*егэ\s*[-–—]?\s*', '', subject)
            subject = re.sub(r'^.*выбору\s*[-–—]?\s*', '', subject)
            subject = subject.lstrip('-–— ').strip()
            if subject and score > 0 and len(subject) > 2:
                result[subject] = score
        return result

    def _normalize_subject(self, subj: str) -> str:
        return self.SUBJECT_ALIASES.get(subj.lower().strip(), subj.lower().strip())

    def match_by_ege(self, user_scores: dict, mode: str = "contract") -> list:
        """
        Подбирает программы по баллам ЕГЭ.
        """
        normalized = {self._normalize_subject(s): sc for s, sc in user_scores.items()}
        ege_col = "eges_contract" if mode == "contract" else "eges_budget"
        results = []

        for _, row in self.df.iterrows():
            ege_text = str(row.get(ege_col, ""))
            if not ege_text or ege_text == "nan":
                continue

            parts = ege_text.lower().split("егэ по выбору")
            mandatory = self._parse_ege_requirements(parts[0] if parts else "")
            choice = self._parse_ege_requirements(parts[1] if len(parts) > 1 else "")

            # Все обязательные предметы есть и ≥ минимума?
            mandatory_ok = all(
                normalized.get(subj, 0) >= min_score
                for subj, min_score in mandatory.items()
            )
            if not mandatory_ok:
                continue

            # Хотя бы один по выбору есть?
            choice_ok = len(choice) == 0 or any(
                normalized.get(subj, 0) >= min_score
                for subj, min_score in choice.items()
            )
            if not choice_ok:
                continue

            required = {**mandatory, **choice}
            total = sum(normalized.get(s, 0) for s in required if s in normalized)

            results.append({
                "program": str(row["program"]), "cost": row.get("cost", "?"),
                "pass_2024": row.get("pass_2024", "?"),
                "budget_2025": row.get("budget_2025", "?"), "total": total,
            })

        results.sort(key=lambda x: x["total"], reverse=True)
        return results

    def format_ege_results(self, results: list, mode: str) -> str:
        """Форматирует результаты подбора по ЕГЭ для Telegram."""
        if not results:
            label = "бюджету" if mode == "budget" else "контракту"
            return f"😔 По вашим баллам не найдено подходящих программ (по {label})."
        label = "бюджет" if mode == "budget" else "контракт"
        lines = [f"🎯 *Подходящие программы ({label}):*\n"]
        for i, r in enumerate(results[:10], 1):
            lines.append(f"*{i}. {r['program']}*")
            lines.append(f"   Сумма ваших баллов: {r['total']}")
            if mode == "budget":
                lines.append(f"   Проходной 2024: {r['pass_2024']}")
                lines.append(f"   Бюджетных мест: {r['budget_2025']}")
            else:
                lines.append(f"   Стоимость: {r['cost']} руб./год")
            lines.append("")
        if len(results) > 10:
            lines.append(f"_...и ещё {len(results) - 10} программ_")
        return "\n".join(lines)

    # Форматирование
    def format_program_info(self, row) -> str:
        if isinstance(row, dict):
            row = pd.Series(row)
        lines = []
        for col, label in self.DISPLAY_COLS.items():
            val = row.get(col)
            if pd.notna(val) and str(val).strip() and str(val).strip().lower() != "nan":
                lines.append(f"{label}: {val}")
        return "\n".join(lines)

    def search_and_format(self, query: str) -> str:
        results = self.find_programs(query)
        if results.empty:
            return ""
        parts = []
        for i, (_, row) in enumerate(results.iterrows()):
            if i >= 5:
                break
            parts.append(f"--- Программа {i+1} ---\n{self.format_program_info(row)}")
        return "\n\n".join(parts)

    def get_all_program_names(self) -> list:
        return sorted(self.df["program"].dropna().unique().tolist())
