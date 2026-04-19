import os
import logging
import telebot
from telebot import types
import config
from pipeline import AssistantPipeline
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Инициализация бота и пайплайна
bot = telebot.TeleBot(config.TELEGRAM_TOKEN)
pipeline = AssistantPipeline()

program_id_cache = {}

user_states = {}

pipeline.initialize()
print("==== Бот запущен ==== ")


def make_confirm_keyboard():
    """Кнопки для подтверждения FAQ-предположения"""
    kb = types.InlineKeyboardMarkup(row_width=2)
    kb.add(
        types.InlineKeyboardButton("✅ Да, это ответ", callback_data="conf_faq_yes"),
        types.InlineKeyboardButton("❌ Нет, другой вопрос", callback_data="conf_faq_no")
    )
    return kb

def make_programs_keyboard(programs: list):
    """Кнопки со списком найденных программ"""
    kb = types.InlineKeyboardMarkup(row_width=1)
    for prog in programs[:6]:
        p_id = str(abs(hash(prog)))[:8]
        program_id_cache[p_id] = prog
        kb.add(types.InlineKeyboardButton(f"📋 {prog[:45]}...", callback_data=f"view_prog|{p_id}"))
    return kb

def make_details_keyboard(prog_id: str):
    """Кнопки действий для конкретной программы"""
    kb = types.InlineKeyboardMarkup(row_width=2)
    kb.add(
        types.InlineKeyboardButton("📊 Баллы", callback_data=f"field|score|{prog_id}"),
        types.InlineKeyboardButton("💰 Цена", callback_data=f"field|cost|{prog_id}"),
        types.InlineKeyboardButton("📝 ЕГЭ", callback_data=f"field|ege|{prog_id}"),
        types.InlineKeyboardButton("🎓 Места", callback_data=f"field|places|{prog_id}")
    )
    kb.add(types.InlineKeyboardButton("📄 Полное описание", callback_data=f"field|full|{prog_id}"))
    return kb

@bot.message_handler(commands=['start'])
def send_welcome(message):
    user_states.pop(message.chat.id, None)
    welcome_text = (
        "**Привет! Я ИИ-помощник, Ольга, абитуриента РАНХиГС.**\n\n"
        "Я помогу найти информацию о программах, проходных баллах, "
        "стоимости обучения и правилах приема.\n\n"
        "*Просто напиши свой вопрос ниже)*"
    )
    bot.send_message(message.chat.id, welcome_text, parse_mode="Markdown")

@bot.message_handler(commands=['reset'])
def reset_state(message):
    user_states.pop(message.chat.id, None)
    bot.send_message(message.chat.id, "🔄 Состояние сброшено. О чем хочешь спросить?")


@bot.message_handler(func=lambda message: True)
def handle_all_messages(message):
    query = message.text.strip()
    chat_id = message.chat.id

    # Показываем статус "печатает"
    bot.send_chat_action(chat_id, 'typing')
    
    result = pipeline.process(query)
    
    scenario = result.get("scenario")
    answer = result.get("answer")
    details = result.get("details", {})

    if scenario == "toxic":
        bot.send_message(chat_id, f"🚫 {answer}")
        return

    if scenario == "FAQ_SUGGEST" and result.get("confidence") < config.FAQ_SIMILARITY_THRESHOLD:
        matched_q = details.get("matched_question", "")
        # Сохраняем оригинальный запрос, чтобы если нажмут "Нет", отправить в LLM
        user_states[chat_id] = {"state": "waiting_faq_confirm", "query": query}
        
        msg = f"💬 Возможно, вы имели в виду вопрос:\n\n*\"{matched_q}\"*?\n\n" \
              f"Если да, нажмите кнопку ниже для получения ответа."
        bot.send_message(chat_id, msg, parse_mode="Markdown", reply_markup=make_confirm_keyboard())
        return

    full_text = answer
    
    programs = details.get("programs", [])
    kb = None
    if programs:
        kb = make_programs_keyboard(programs)

    try:
        bot.send_message(chat_id, full_text, parse_mode="Markdown", reply_markup=kb)
    except Exception as e:
        # Fallback если Markdown сломался
        bot.send_message(chat_id, full_text, reply_markup=kb)


@bot.callback_query_handler(func=lambda call: True)
def handle_callbacks(call):
    chat_id = call.message.chat.id
    data = call.data

    # Подтверждение FAQ
    if data == "conf_faq_yes":
        bot.answer_callback_query(call.id)
        # Получаем ответ через пайплайн повторно (уже точно)
        state = user_states.pop(chat_id, {})
        query = state.get("query", "")
        # Чтобы не делать двойной поиск, можно было бы сохранить ответ в state, 
        # но для простоты прогоним еще раз или вызовем напрямую
        result = pipeline.process(query)
        bot.edit_message_text(result["answer"], chat_id,
                              call.message.message_id, parse_mode="Markdown")

    elif data == "conf_faq_no":
        bot.answer_callback_query(call.id, "Переключаю на ИИ...")
        state = user_states.pop(chat_id, {})
        query = state.get("query", "")
        # FIX B: Используем метод МИНУЯ FAQ
        result = pipeline.process_llm_only(query)
        bot.edit_message_text("FAQ не подошел. Ищу информацию в базе знаний...", 
                             chat_id, call.message.message_id)
        bot.send_message(chat_id, result["answer"], parse_mode="Markdown")

    # Просмотр конкретной программы из списка
    elif data.startswith("view_prog|"):
        p_id = data.split("|")[1]
        prog_name = program_id_cache.get(p_id, "Программа")
        bot.answer_callback_query(call.id)
        bot.send_message(chat_id, f"📍 **{prog_name}**\nЧто именно вас интересует?", 
                         parse_mode="Markdown", reply_markup=make_details_keyboard(p_id))

    # Получение конкретного поля
    elif data.startswith("field|"):
        _, field, p_id = data.split("|")
        prog_name = program_id_cache.get(p_id)
        
        if not prog_name:
            bot.answer_callback_query(call.id, "Ошибка: данные устарели")
            return

        bot.answer_callback_query(call.id)
        # Используем внутренний поиск программ
        field_answer = pipeline.program_search.get_program_field(prog_name, field)
        bot.send_message(chat_id, field_answer, parse_mode="Markdown")

if __name__ == "__main__":
    try:
        bot.infinity_polling()
    except Exception as e:
        logger.error(f"Ошибка бота: {e}")