import os
import logging
import httpx
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO)

BOT_TOKEN = os.environ.get("BOT_TOKEN", "8949573765:AAFE3XGvRI2pNDeWrxKUpHlOH7YMa17qurA")
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "tick-bot-secret")
RENDER_URL = os.environ.get("RENDER_EXTERNAL_URL", "")

app = Flask(__name__)

from model import predict


def telegram_api(method, data=None):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/{method}"
    r = httpx.post(url, json=data, timeout=30)
    return r.json()


def send_message(chat_id, text):
    telegram_api("sendMessage", {"chat_id": chat_id, "text": text})


def download_photo(file_id):
    info = telegram_api("getFile", {"file_id": file_id})
    if not info.get("ok"):
        return None
    file_path = info["result"]["file_path"]
    url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"
    r = httpx.get(url, timeout=30)
    return r.content


@app.route("/", methods=["GET"])
def health():
    return "Bot is running"


@app.route(f"/{WEBHOOK_SECRET}", methods=["POST"])
def webhook():
    data = request.get_json(force=True)

    if "message" not in data:
        return jsonify({"ok": True})

    message = data["message"]
    chat_id = message["chat"]["id"]

    if "text" in message and message["text"] == "/start":
        send_message(chat_id, "Привет! Отправь мне фото жука, и я определю, клещ это или нет.")

    elif "photo" in message:
        send_message(chat_id, "Анализирую фото...")
        file_id = message["photo"][-1]["file_id"]
        image_bytes = download_photo(file_id)

        if image_bytes:
            result = predict(image_bytes)
            if result["has_tick"]:
                conf = result["confidence"] * 100
                count = len(result["detections"])
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
            send_message(chat_id, text)
        else:
            send_message(chat_id, "Не удалось загрузить фото. Попробуй ещё раз.")

    return jsonify({"ok": True})


if RENDER_URL and WEBHOOK_SECRET:
    webhook_url = f"{RENDER_URL}/{WEBHOOK_SECRET}"
    result = telegram_api("setWebhook", {"url": webhook_url})
    logging.info(f"Webhook set: {result}")
