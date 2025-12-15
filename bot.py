import asyncio
import io
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Awaitable
import html

from dotenv import load_dotenv

import numpy as np
from PIL import Image, UnidentifiedImageError
import onnxruntime as ort

from aiogram import Bot, Dispatcher, F, Router
from aiogram.types import Message
from aiogram.filters import CommandStart, Command

load_dotenv()

# =========================
# CONFIG
# =========================
BOT_TOKEN = os.getenv("BOT_TOKEN")

ACCESS_KEY = os.getenv("ACCESS_KEY")

LANGUAGE = os.getenv("LANGUAGE", "ru").lower()

AUTH_DB_PATH = Path("authorized_users.json")

ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "dog_breed_efficientnet_b0.onnx"
META_PATH = ARTIFACTS_DIR / "dog_breed_meta.json"

MESSAGES_PATH = ARTIFACTS_DIR / "messages.json"
with open(MESSAGES_PATH, "r", encoding="utf-8") as f:
    MESSAGES_ALL = json.load(f)

BREEDS_PATH = ARTIFACTS_DIR / f"breeds_{LANGUAGE}.json"
LANGUAGE_FOUND = True

if not BREEDS_PATH.exists():
    if LANGUAGE != "ru":
        print(f"Warning: breeds_{LANGUAGE}.json not found. Falling back to 'ru'.")
    LANGUAGE_FOUND = False

if LANGUAGE not in MESSAGES_ALL:
    if LANGUAGE_FOUND and LANGUAGE != "ru":
        print(f"Warning: Language '{LANGUAGE}' not found in messages.json. Falling back to 'ru'.")
    LANGUAGE_FOUND = False

if not LANGUAGE_FOUND:
    LANGUAGE = "ru"
    BREEDS_PATH = ARTIFACTS_DIR / "breeds_ru.json"

with open(BREEDS_PATH, "r", encoding="utf-8") as f:
    BREEDS = json.load(f)

MESSAGES = MESSAGES_ALL.get(LANGUAGE, MESSAGES_ALL["ru"])

def get_translation(key: str, **kwargs) -> str:
    """Get translated message with parameter substitution."""
    text = MESSAGES.get(key, key)
    if kwargs:
        try:
            text = text.format(**kwargs)
        except KeyError:
            pass
    return text

def label_to_lang(full_label: str) -> str:
    """Convert model label (e.g., 'animal dog golden_retriever') to localized breed name."""
    key = full_label.replace("animal dog ", "").strip()
    return BREEDS.get(key, key.replace("_", " ").strip())


WORKERS = int(os.getenv("WORKERS"))

ALBUM_FLUSH_DELAY_SEC = 1.0

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# =========================
# AUTH STORAGE
# =========================
def load_authorized_users() -> set[int]:
    if not AUTH_DB_PATH.exists():
        return set()
    try:
        with open(AUTH_DB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(int(x) for x in data.get("users", []))
    except Exception:
        return set()

def save_authorized_users(users: set[int]) -> None:
    with open(AUTH_DB_PATH, "w", encoding="utf-8") as f:
        json.dump({"users": sorted(list(users))}, f, ensure_ascii=False, indent=2)

AUTHORIZED_USERS = load_authorized_users()

def is_authorized(user_id: int) -> bool:
    return user_id in AUTHORIZED_USERS

def authorize(user_id: int) -> None:
    AUTHORIZED_USERS.add(user_id)
    save_authorized_users(AUTHORIZED_USERS)

# =========================
# MODEL LOADING (ONNX)
# =========================
def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)

with open(META_PATH, "r", encoding="utf-8") as f:
    META = json.load(f)

IMAGE_SIZE = int(META["image_size"])
IDX_TO_CLASS = {int(k): v for k, v in META["idx_to_class"].items()}

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = 1
sess_opts.inter_op_num_threads = 1

SESSION = ort.InferenceSession(
    MODEL_PATH.as_posix(),
    sess_options=sess_opts,
    providers=["CPUExecutionProvider"],
)
INPUT_NAME = SESSION.get_inputs()[0].name

def preprocess_pil(img: Image.Image) -> np.ndarray:
    """Preprocess PIL image for ONNX model: normalize and convert HWC -> NCHW format."""
    img = img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, 0)
    return arr.astype(np.float32)

def predict_one_image_bytes(image_bytes: bytes, top_k: int = 1) -> list[tuple[str, float]]:
    img = Image.open(io.BytesIO(image_bytes))
    x = preprocess_pil(img)
    logits = SESSION.run(None, {INPUT_NAME: x})[0]
    probs = softmax(logits)[0]

    top_k = min(top_k, probs.shape[0])
    top_idx = np.argsort(-probs)[:top_k]
    return [(IDX_TO_CLASS[int(i)], float(probs[int(i)])) for i in top_idx]

# =========================
# QUEUE: batch per user-message/album
# =========================
@dataclass
class Job:
    chat_id: int
    reply_to_message_id: int
    images: list[bytes]
    created_at: float

JOB_QUEUE: asyncio.Queue[Job] = asyncio.Queue()

async def run_inference_batch(images: list[bytes]) -> list[tuple[str, float]]:
    results: list[tuple[str, float]] = []
    for img_bytes in images:
        preds = await asyncio.to_thread(predict_one_image_bytes, img_bytes, 1)
        results.append(preds[0])
    return results

async def worker_loop(bot: Bot, worker_id: int) -> None:
    while True:
        job = await JOB_QUEUE.get()
        try:
            preds = await run_inference_batch(job.images)
            lines = []
            for i, (label, prob) in enumerate(preds, start=1):
                prob_percent = f"{prob*100:.1f}"
                if prob < 0.25:
                    lines.append(get_translation("result_dog_not_found", i=i, prob=prob_percent))
                else:
                    breed = label_to_lang(label)
                    lines.append(get_translation("result_image_prefix", i=i, breed=breed, prob=prob_percent))

            text = "\n".join(lines)
            await bot.send_message(
                chat_id=job.chat_id,
                text=html.escape(text),
                reply_to_message_id=job.reply_to_message_id,
                parse_mode="HTML",
            )

        except Exception as e:
            await bot.send_message(
                chat_id=job.chat_id,
                text=get_translation("error_recognition", error=str(e)),
                reply_to_message_id=job.reply_to_message_id,
            )
        finally:
            JOB_QUEUE.task_done()

# =========================
# ALBUM COLLECTOR (media_group_id)
# =========================
class AlbumCollector:
    def __init__(self):
        self._albums: dict[str, list[Message]] = {}
        self._flush_tasks: dict[str, asyncio.Task] = {}
        self._processing_callbacks: dict[str, Callable[[list[Message]], Awaitable[None]]] = {}

    def add(self, message: Message, process_callback: Callable[[list[Message]], Awaitable[None]]) -> None:
        mgid = str(message.media_group_id)
        self._albums.setdefault(mgid, []).append(message)
        self._processing_callbacks[mgid] = process_callback

        if mgid not in self._flush_tasks:
            self._flush_tasks[mgid] = asyncio.create_task(self._flush_later(mgid))

    async def _flush_later(self, mgid: str) -> None:
        await asyncio.sleep(ALBUM_FLUSH_DELAY_SEC)
        self._flush_tasks.pop(mgid, None)
        if mgid in self._albums:
            messages = self._albums.pop(mgid, None)
            callback = self._processing_callbacks.pop(mgid, None)
            if messages and callback:
                await callback(messages)

    def pop_if_ready(self, mgid: str) -> list[Message] | None:
        if mgid in self._flush_tasks:
            return None
        self._processing_callbacks.pop(mgid, None)
        return self._albums.pop(mgid, None)

ALBUMS = AlbumCollector()

# =========================
# BOT HANDLERS
# =========================
router = Router()

def start_deeplink_text() -> str:
    """Get Telegram deep link info text (format: https://t.me/<bot_username>?start=<ACCESS_KEY>)."""
    return get_translation("start_deeplink_info")

@router.message(CommandStart())
async def cmd_start(message: Message):
    user_id = message.from_user.id if message.from_user else 0

    text = message.text or ""
    parts = text.split(maxsplit=1)
    payload = parts[1].strip() if len(parts) > 1 else ""

    if is_authorized(user_id):
        await message.answer(get_translation("start_already_authorized"))
        return

    if payload == ACCESS_KEY:
        authorize(user_id)
        await message.answer(get_translation("start_access_granted"))
        return

    deeplink_info = start_deeplink_text()
    await message.answer(
        get_translation("start_welcome", deeplink_info=deeplink_info),
        parse_mode="Markdown",
    )

@router.message(Command("help"))
async def cmd_help(message: Message):
    await message.answer(get_translation("help_commands"))

@router.message(Command("logout"))
async def cmd_logout(message: Message):
    user_id = message.from_user.id if message.from_user else 0
    if user_id in AUTHORIZED_USERS:
        AUTHORIZED_USERS.remove(user_id)
        save_authorized_users(AUTHORIZED_USERS)
        await message.answer(get_translation("logout_success"))
    else:
        await message.answer(get_translation("logout_not_authorized"))

@router.message(F.text)
async def on_text(message: Message):
    user_id = message.from_user.id if message.from_user else 0

    if not is_authorized(user_id):
        if (message.text or "").strip() == ACCESS_KEY:
            authorize(user_id)
            await message.answer(get_translation("start_access_granted"))
        else:
            await message.answer(get_translation("text_wrong_key"))
        return

    await message.answer(get_translation("text_send_image"))

async def download_photo_bytes(bot: Bot, message: Message) -> bytes:
    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    data = await bot.download_file(file.file_path)
    return data.read()

async def download_document_bytes(bot: Bot, message: Message) -> tuple[bytes, str]:
    doc = message.document
    filename = doc.file_name or ""
    file = await bot.get_file(doc.file_id)
    data = await bot.download_file(file.file_path)
    return data.read(), filename

def is_openable_by_pil(image_bytes: bytes) -> bool:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False

def normalize_to_png_bytes(image_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True)
    return out.getvalue()

async def enqueue_job(message: Message, images: list[bytes], bot: Bot) -> None:
    await JOB_QUEUE.put(
        Job(
            chat_id=message.chat.id,
            reply_to_message_id=message.message_id,
            images=images,
            created_at=time.time(),
        )
    )

    qpos = JOB_QUEUE.qsize()
    if len(images) == 1:
        await message.reply(get_translation("queue_added_single", qpos=qpos))
    else:
        await message.reply(get_translation("queue_added_multiple", count=len(images), qpos=qpos))

async def process_album(messages: list[Message], bot: Bot) -> None:
    """Process completed album (media group) by downloading and normalizing all images."""
    if not messages:
        return
    
    images: list[bytes] = []
    for msg in messages:
        b = await download_photo_bytes(bot, msg)
        b = normalize_to_png_bytes(b)
        images.append(b)

    await enqueue_job(messages[0], images, bot)

@router.message(F.photo)
async def on_photo(message: Message, bot: Bot):
    user_id = message.from_user.id if message.from_user else 0
    if not is_authorized(user_id):
        await message.answer(get_translation("photo_need_key"), parse_mode="Markdown")
        return

    if message.media_group_id:
        async def callback(messages: list[Message]) -> None:
            await process_album(messages, bot)
        
        ALBUMS.add(message, callback)
        return

    image_bytes = await download_photo_bytes(bot, message)
    image_bytes = normalize_to_png_bytes(image_bytes)
    await enqueue_job(message, [image_bytes], bot)

@router.message(F.document)
async def on_document(message: Message, bot: Bot):
    user_id = message.from_user.id if message.from_user else 0
    if not is_authorized(user_id):
        await message.answer(get_translation("photo_need_key"), parse_mode="Markdown")
        return

    file_bytes, filename = await download_document_bytes(bot, message)
    ext = Path(filename).suffix.lower()

    if ext and ext in ALLOWED_EXTENSIONS:
        try:
            file_bytes = normalize_to_png_bytes(file_bytes)
            await enqueue_job(message, [file_bytes], bot)
            return
        except Exception:
            await message.reply(get_translation("document_read_error"))
            return

    if is_openable_by_pil(file_bytes):
        try:
            file_bytes = normalize_to_png_bytes(file_bytes)
            await enqueue_job(message, [file_bytes], bot)
        except Exception:
            await message.reply(get_translation("document_convert_error"))
        return

    await message.reply(get_translation("document_unsupported_format"))

# =========================
# MAIN
# =========================
async def main():
    if BOT_TOKEN == "PASTE_YOUR_BOT_TOKEN_HERE":
        raise RuntimeError(get_translation("error_bot_token"))

    bot = Bot(BOT_TOKEN)
    dp = Dispatcher()
    dp.include_router(router)

    for i in range(WORKERS):
        asyncio.create_task(worker_loop(bot, i))

    print(f"Bot started. Workers={WORKERS}. ImageSize={IMAGE_SIZE}. Language={LANGUAGE}. AuthorizedUsers={len(AUTHORIZED_USERS)}")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
