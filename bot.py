import os
import logging
from flask import Flask, request, jsonify
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

logging.basicConfig(level=logging.INFO)

BOT_TOKEN = os.environ.get("BOT_TOKEN", "8949573765:AAFE3XGvRI2pNDeWrxKUpHlOH7YMa17qurA")
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "tick-bot-secret")

app = Flask(__name__)

application = ApplicationBuilder().token(BOT_TOKEN).build()


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


application.add_handler(CommandHandler("start", start))
application.add_handler(MessageHandler(filters.PHOTO, handle_photo))


@app.route("/", methods=["GET"])
def health():
    return "Bot is running"


@app.route(f"/{WEBHOOK_SECRET}", methods=["POST"])
def webhook():
    data = request.get_json(force=True)
    update = Update.de_json(data, application.bot)
    import asyncio
    asyncio.run(application.process_update(update))
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    render_url = os.environ.get("RENDER_EXTERNAL_URL", "https://tick-bot-s50p.onrender.com")

    import asyncio
    asyncio.run(application.initialize())
    asyncio.run(application.start())

    webhook_url = f"{render_url}/{WEBHOOK_SECRET}"
    asyncio.run(application.bot.set_webhook(url=webhook_url))
    logging.info(f"Webhook set to: {webhook_url}")

    app.run(host="0.0.0.0", port=port)
