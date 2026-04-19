import config
from data_loader import load_stop_words
from toxicity_filter import ToxicityFilter
from embedding_engine import EmbeddingEngine
from program_search import ProgramSearchEngine, expand_query, SYNONYMS
from llm_client import LLMClient
import re
from natasha import MorphVocab, NewsMorphTagger, Doc, NewsEmbedding, Segmenter
import db_manager
import pandas as pd
from sentence_transformers import CrossEncoder


DATA_KEYWORDS = [
    "стоимость", "стоит", "цена", "балл", "проходн", "бюджет", "мест",
    "егэ", "экзамен", "платн", "контракт", "форма обучения", "срок",
    "трек", "направлени", "сколько стоит", "какие егэ", "какие экзамены",
    "оплат", "денег", "дорог", "дёшев", "дешев", "руб", "квот", "предмет",
    "дешевле", "дороже", "самая дорогая", "самая дешёвая", "самая дешевая",
    "сравни", "сравнение", "отличается", "лучше", "хуже",
    "все программы", "покажи все", "список программ",
    "больше всего мест", "меньше всего",
]


class AssistantPipeline:
    def __init__(self):
        self.segmenter = Segmenter()
        self.emb = NewsEmbedding()
        self.morph_vocab = MorphVocab()
        self.morph_tagger = NewsMorphTagger(self.emb)
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return

        db_manager.rebuild_db(force=False)

        stop_words = load_stop_words()

        self.toxicity_filter = ToxicityFilter(stop_words)

        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.embedding = EmbeddingEngine()

        # Индекс FAQ
        faq_data = db_manager.get_all_faq()
        faq_questions = [self.clean_text(r["Question"]) for r in faq_data]
        faq_meta = [
            {"Question": r["Question"],
             "Answer": r["Answer"],
             "Question type": r.get("Question type", "")}
            for r in faq_data
            ]
        
        self.embedding.build_index("faq", faq_questions, faq_meta)

        # Индекс базы знаний
        kb_data = db_manager.get_all_knowledge()
        kb_chunks = [f"{r['header']}\n{r['text']}" for r in kb_data]
        self.embedding.build_index("kb", kb_chunks)

        # Индекс программ
        prog_data = db_manager.get_all_programs()
        prog_texts = []
        for row in prog_data:
            institute_full = str(row.get("institute", "")).lower()
            alias = next((k for k, v in SYNONYMS.items() if v in institute_full), "")
            t = " ".join([
                str(row.get("program", "")),
                str(row.get("megacluster", "")),
                str(row.get("major", "")),
                institute_full,
                alias,
                str(row.get("tracks", "")),
                str(row.get("desc", ""))[:200],
                str(row.get("skills", ""))[:200],
            ])
            prog_texts.append(t)
        self.embedding.build_index("programs", prog_texts, prog_data)

        programs_df = pd.DataFrame(prog_data)
        for col in ["program", "megacluster", "institute", "major", "tracks"]:
            if col in programs_df.columns:
                programs_df[col + "_lower"] = programs_df[col].astype(str).str.lower().str.strip()
        self.program_search = ProgramSearchEngine(programs_df)

        self.llm = LLMClient()
        self._initialized = True


    def process(self, query: str) -> dict:
        # ШАГ 1: Токсичность
        if self.toxicity_filter.is_toxic(query):
            return self._result(config.TOXICITY_RESPONSE, "toxic")

        # ШАГ 2: FAQ - точное совпадение
        faq_hits = self.embedding.search("faq", self.clean_text(query), top_k=config.FAQ_TOP_K)

        if faq_hits:
            cosin, _, top_meta = faq_hits[0] 
            if cosin > config.FAQ_SIMILARITY_THRESHOLD:
                return self._result(
                    top_meta["Answer"], "FAQ_SIMILARITY",
                    matched_question=top_meta["Question"],
                    question_type=top_meta.get("Question type", "")
                )
            
            elif cosin > config.FAQ_SUGGEST_THRESHOLD:
                return self._result(
                        top_meta["Answer"],"FAQ_SUGGEST", 
                        matched_question=top_meta["Question"],
                        question_type=top_meta.get("Question type", "")
                    )
                
        # ШАГ 3: LLM
        return self._to_llm(query, faq_hits)


    def process_llm_only(self, query: str) -> dict:
        """
        Обрабатывает запрос, игнорируя FAQ-совпадения. 
        Вызывается когда пользователь нажал "Нет, другое" на FAQ-suggest.
        Вопрос идёт напрямую в LLM
        """
        # Токсичность всё равно проверяем
        if self.toxicity_filter.is_toxic(query):
            return self._result(config.TOXICITY_RESPONSE, "toxic")

        clean_query = self.clean_text(query)
        faq_hits = self.embedding.search("faq", clean_query, top_k=3) # нужно для создания контекста
        return self._to_llm(query, faq_hits)

    # Логика LLM (SQL + RAG)
    def _to_llm(self, query: str, faq_hits: list) -> dict:
        # Собираем контексты из всех источников
        program_ctx = self._build_program_context(query)
        kb_ctx = self._build_kb_context(query)
        faq_ctx = self._build_faq_context(faq_hits)

        # Проверка на наличие информации
        if not kb_ctx and not program_ctx and not faq_ctx:
            return self._result(config.NOT_FOUND_RESPONSE, "not_found")
        
        # Классифицируем
        query_type = self._classify(query, program_ctx, kb_ctx, faq_ctx)

        if query_type == "program_data":
            return self._answer_program_sql(query)
        return self._answer_rag(query, kb_ctx, program_ctx, faq_ctx)

    # Классификация
    def _classify(self, query: str, program_ctx: str, kb_ctx: str, faq_ctx: str) -> str:
        q = query.lower()
        # У прощаем классификацию, если в запросе явно есть слова, 
        # Связанные с данными программ (стоимость, баллы, экзамены и т.п.)
        if any(kw in q for kw in DATA_KEYWORDS):
            return "program_data"
        if not program_ctx:
            return "general"
        return self.llm.classify_query(query, program_ctx, kb_ctx, faq_ctx)

    # SQL-AGENT
    def _answer_program_sql(self, query: str) -> dict:
        sql = self.llm.generate_sql(query)
        if not sql:
            return self._answer_program_fallback(query)
        try:
            rows = db_manager.run_safe_select(sql)
        except Exception as e:
            return self._answer_program_fallback(query)
        if not rows:
            return self._answer_program_fallback(query)

        answer = self.llm.format_sql_answer(query, rows, sql)

        programs_found = []
        for row in rows:
            name = row.get("program", "")
            if name and name.lower() not in ("nan", "") and name not in programs_found:
                programs_found.append(name)

        return self._result(
            answer, "SQL",
            sql=sql,
            rows_count=len(rows),
            programs=programs_found,
        )

    def _answer_program_fallback(self, query: str) -> dict:
        ctx = self._build_program_context(query)

        if self.llm.is_available():
            user_msg = (
                f"Вопрос абитуриента: {query}\n\n"
                f"Данные из таблицы образовательных программ:\n{ctx}\n\n"
                f"Дай точный ответ на основе этих данных."
            )
            answer = self.llm.generate(config.SYSTEM_PROMPT_SQL, user_msg)
        else:
            answer = f"Вот что я нашёл:\n\n{ctx}"

        # ищем программы в контексте для кнопок
        emb_hits = self.embedding.search("programs", query, top_k=3)
        programs_found = []
        for _, _, meta in emb_hits:
            name = meta.get("program", "") if isinstance(meta, dict) else ""
            if name and name not in programs_found:
                programs_found.append(name)

        return self._result(answer, "NoSQL", programs=programs_found)

    # RAG
    def _answer_rag(self, query: str, kb_ctx: str, program_ctx: str, faq_ctx: str) -> dict:
        """
        RAG-ответ с контекстом из всех источников
        """
        # Собираем комбинированный контекст
        parts = []
        if faq_ctx:
            parts.append(f"Ответы из FAQ (похожие вопросы):\n{faq_ctx}")
        if kb_ctx:
            parts.append(f"Из базы знаний:\n{kb_ctx}")
        if program_ctx:
            parts.append(f"Из таблицы программ:\n{program_ctx}")
        combined = "\n\n".join(parts)

        if self.llm.is_available():
            user_msg = (
                f"Вопрос абитуриента: {query}\n\n"
                f"Контекст:\n{combined}\n\n"
                f"Ответь на вопрос на основе контекста. "
                f"Если в FAQ есть подходящий ответ - используй его."
                f"Если в запросе присутствует название института, произведи поиск по всем возможным сокращениям."
            )
            answer = self.llm.generate(config.SYSTEM_PROMPT_RAG, user_msg)
        else:
            answer = f"Вот что я нашёл:\n\n{combined[:500]}..."

        return self._result(answer, "RAG")

    # Сбор контекста для LLM
    def _build_program_context(self, query: str) -> str:
        expanded = expand_query(query)
        db_results = db_manager.search_programs(expanded, limit=5)
        ctx = db_manager.format_results(db_results)

        emb_hits = self.embedding.search("programs", query, top_k=3)
        for score, _, meta in emb_hits:
            if score > 0.4:
                info = db_manager.format_program(meta)
                if info not in ctx:
                    ctx += f"\n\n Дополнительно \n{info}"
        return ctx

    def _build_kb_context(self, query: str) -> str:
        kb_hits = self.embedding.search("kb", query, top_k=config.RAG_TOP_K)
        relevant = [(s, t) for s, t, _ in kb_hits]
        if not relevant:
            return ""
        return "\n\n".join([text for _, text in relevant])

    def _build_faq_context(self, faq_hits: list) -> str:
        """
        Берём top FAQ-совпадения с score > 0.3 и передаём в LLM
        """
        if not faq_hits:
            return ""
        parts = []
        for score, _, meta in faq_hits[:5]:  # берём до 5 ближайших
            if score > 0.3:
                parts.append(f"Вопрос: {meta['Question']}\nОтвет: {meta['Answer']}")
        return "\n\n".join(parts)

    
    @staticmethod
    def _result(answer, scenario, **details):
        return {
            "answer": answer,
            "scenario": scenario,
            "details": details,
        }

    def clean_text(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[«»"\']', ' ', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'[—–-]+', '-', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
