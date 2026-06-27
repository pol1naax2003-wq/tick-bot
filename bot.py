import os
import httpx
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

API_URL = "http://127.0.0.1:8000/predict"
BOT_TOKEN = os.environ.get("BOT_TOKEN", "8949573765:AAFE3XGvRI2pNDeWrxKUpHlOH7YMa17qurA")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Отправь мне фото жука, и я определю, клещ это или нет."
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1]
    file = await photo.get_file()
    image_bytes = await file.download_as_bytearray()

    from model import predict
    data = predict(bytes(image_bytes))

    if data["has_tick"]:
        conf = data["confidence"] * 100
        count = len(data["detections"])
        text = (
            f"На фото обнаружен клещ!\n"
            f"Уверенность: {conf:.1f}%\n"
            f"Количество: {count}\n\n"
            f"Рекомендации:\n"
            f"- Не удаляйте клеща самостоятельно\n"
            f"- Обратитесь в ближайшее медучреждение\n"
            f"- Обработайте место укуса антисептиком"
        )
    else:
        text = "Клещ не обнаружен. Похоже, это просто жук."

    await update.message.reply_text(text)


def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    print("Бот запущен...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
