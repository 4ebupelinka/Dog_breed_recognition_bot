# Dog Breed Classifier Telegram Bot (ONNX Runtime)

A study project: a Telegram bot that recognizes dog breeds from images using a fine-tuned EfficientNet model exported to ONNX for fast and lightweight CPU inference (suitable for VPS without GPU).

## Features

- **CPU inference with ONNX Runtime** (no GPU required)
- **Job queue + workers** (set workers according to your vCPU count)
- Accepts images as **Telegram photos** or **documents**
- **Silent conversion** of supported/convertible image files on the server side
- Supports **albums (multiple images in one message)** and returns a single consolidated reply:
  - `Image 1 — ...`
  - `Image 2 — ...`
- **Multi-language support**:
  - Bot messages and breed names are translated based on the `LANGUAGE` setting in `.env`
  - Languages are automatically detected from available translation files
  - Default language: `ru` (Russian)
  - To add a new language, simply create the translation files (no code changes needed)
- **Access key gating** for study usage:
  - send the key as a message, or
  - use deep-link: `/start <ACCESS_KEY>`

> Note: Access-key gating is meant to prevent accidental usage by random users. If someone obtains the deep-link/QR with the key, they can authorize.

---

## Model

### Architecture (trained)
- **Backbone**: EfficientNet-B0 (transfer learning)
- **Input**: RGB image (most are around **224×224**, but the dataset reports `image_size = "nochange"`)
- **Preprocessing**: resize to the model input size + ImageNet normalization
- **Head**: final classification layer replaced with `Linear(in_features, num_classes)`
- **Deployment**: exported to **ONNX**, optionally quantized to **INT8** for CPU efficiency

---

## Dataset

This project uses a dataset provided by **www.images.cv**.

Summary:

- **Total images**: **19,921**
- **Labels**: `["dog"]` (the dataset is organized into breed folders under train/val/test)
- **Color mode**: **color**
- **Image size**: `"nochange"` (varies; the model resizes internally for inference)
- **Splits**:
  - **train**: 11,991
  - **val**: 1,921
  - **test**: 6,009
- **Folder structure**: `data/train`, `data/val`, `data/test`
- **Split ratio**: **60% / 10% / 30%** (train/val/test)
- **Augmentation**: disabled in the provided dataset metadata
  - augmented images would have `_aug` suffix in the train folder (not used here)

Useful links (from the dataset metadata):
- `images.cv/how-to-use`
- `images.cv/faq`
- `images.cv/computer-vision-dataset-categories`
- `images.cv/computer-vision-image-datasets-index`
- `images.cv/contact`

---

## Project structure

```

.
├─ bot.py
├─ requirements.txt
├─ .env                  # NOT committed
├─ artifacts/
│  ├─ dog_breed_efficientnet_b0.onnx
│  ├─ dog_breed_meta.json
│  ├─ breeds_ru.json      # Russian breed names mapping
│  ├─ breeds_eng.json     # English breed names mapping
│  └─ messages.json       # Bot messages translations (ru/eng)
└─ authorized_users.json  # generated at runtime

````

---

## Requirements

- Python **3.10+**
- A Telegram Bot Token from **@BotFather**

---

## Installation (Windows)

### 1) Clone the repo
```powershell
git clone https://github.com/4ebupelinka/Dog_breed_recognition_bot
cd Dog_breed_recognition_bot
````

### 2) Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1
```

### 3) Install dependencies

```powershell
python -m pip install -U pip
pip install -r requirements.txt
```

### 4) Create `.env`

Create a `.env` file next to `bot.py`:

```env
BOT_TOKEN=PASTE_YOUR_TELEGRAM_BOT_TOKEN
ACCESS_KEY=MY_STUDY_KEY_123
WORKERS=2
LANGUAGE=ru
```

Language examples:
- `ru` - Russian (default)
- `eng` - English

> Note: Any language can be added by creating the appropriate translation files. See the "Adding a New Language" section below.

### 5) Run the bot

```powershell
python bot.py
```

---

## Installation (Ubuntu / VPS)

### 1) System setup

```bash
sudo apt update && sudo apt -y upgrade
sudo apt install -y python3 python3-venv python3-pip git
```

### 2) Clone and install

```bash
git clone https://github.com/4ebupelinka/Dog_breed_recognition_bot
cd Dog_breed_recognition_bot
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 3) Create `.env`

```bash
nano .env
```

Example:

```env
BOT_TOKEN=PASTE_YOUR_TELEGRAM_BOT_TOKEN
ACCESS_KEY=MY_STUDY_KEY_123
WORKERS=2
LANGUAGE=ru
```

Language examples:
- `ru` - Russian (default)
- `eng` - English

> Note: Any language can be added by creating the appropriate translation files. See the "Adding a New Language" section below.

Protect the file:

```bash
chmod 600 .env
```

### 4) Run

```bash
source venv/bin/activate
python bot.py
```

---

## Run as a systemd service (Ubuntu)

Create the service:

```bash
sudo nano /etc/systemd/system/Dog_breed_recognition_bot.service
```

Paste (replace `<Dog_breed_recognition_bot folder location>` and <YOUR_USER>):

```ini
[Unit]
Description=Dog Breed Telegram Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=<YOUR_USER>
WorkingDirectory=<Dog_breed_recognition_bot folder location>/Dog_breed_recognition_bot
EnvironmentFile=<Dog_breed_recognition_bot folder location>/Dog_breed_recognition_bot/.env
ExecStart=<Dog_breed_recognition_bot folder location>/Dog_breed_recognition_bot/venv/bin/python /<Dog_breed_recognition_bot folder location>/Dog_breed_recognition_bot/bot.py
Restart=always
RestartSec=3
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable Dog_breed_recognition_bot
sudo systemctl start Dog_breed_recognition_bot
sudo systemctl status Dog_breed_recognition_bot --no-pager
journalctl -u Dog_breed_recognition_bot -f
```

---

## Usage

1. Open the bot and run:

   * `/start`
2. Provide the access key (either send it as a message or use `/start <ACCESS_KEY>`).
3. Send an image:

   * as a Telegram photo, or
   * as a document (the bot validates extension and/or attempts conversion).
4. If you send an album, the bot will queue all images and respond with one consolidated result message.

---

## Notes

* For best CPU performance on VPS, prefer the **INT8** ONNX model.
* If your VPS has `N` vCPUs, set `WORKERS=N` (or `N-1`) in `.env`.
* The bot supports multiple languages via the `LANGUAGE` environment variable. All bot messages and breed names will be displayed in the selected language. The bot automatically detects available languages from translation files in the `artifacts/` directory. If the specified language is not found or `LANGUAGE` is not set, Russian (`ru`) is used as the default. A warning message will be printed to the console if translation files for the requested language are missing.

---

## Adding a New Language

To add support for a new language (e.g., `de` for German), **no code changes are required**. Simply create the translation files:

1. **Create a breed names file**: Create `artifacts/breeds_<lang>.json` (e.g., `breeds_de.json`) with the same structure as `breeds_ru.json`. Map each breed key to its translated name:
   ```json
   {
     "golden_retriever": "Golden Retriever",
     "german_sheperd": "Deutscher Schäferhund",
     "beagle": "Beagle",
     ...
   }
   ```
   
   > **Important**: Include translations for **all** breed keys that exist in `breeds_ru.json`.

2. **Add translations to messages.json**: In `artifacts/messages.json`, add a new section with your language code. **You must translate ALL message keys** used by the bot:
   ```json
   {
     "ru": {...},
     "eng": {...},
     "de": {
       "start_already_authorized": "Du bist bereits autorisiert. Sende ein Bild (Foto oder Datei) 🙂",
       "start_access_granted": "Zugriff gewährt ✅\nSende ein Bild (Foto oder Datei).",
       "start_welcome": "Hallo! Ein Zugriffsschlüssel wird benötigt...",
       "start_deeplink_info": "Du kannst den Bot mit einem Schlüssel starten (Deep-Link)...",
       "help_commands": "Befehle:\n/start — Start\n/help — Hilfe...",
       "logout_success": "Ok, Zugriff entfernt...",
       "logout_not_authorized": "Du bist ohnehin nicht autorisiert.",
       "text_wrong_key": "Falscher Schlüssel. Versuche es erneut.",
       "text_send_image": "Sende ein Bild (Foto oder Datei).",
       "photo_need_key": "Zuerst wird ein Zugriffsschlüssel benötigt...",
       "document_read_error": "Konnte Bild aus Datei nicht lesen...",
       "document_convert_error": "Konnte Datei nicht korrekt konvertieren...",
       "document_unsupported_format": "Nicht unterstütztes Dateiformat...",
       "queue_added_single": "Empfangen. Zur Warteschlange hinzugefügt...",
       "queue_added_multiple": "Empfangen {count} Bilder. Zur Warteschlange hinzugefügt...",
       "result_image_prefix": "Bild {i} — {breed} ({prob}%)",
       "error_recognition": "Erkennungsfehler: {error}",
       "error_bot_token": "Fülle BOT_TOKEN in Umgebungsvariable oder im Code aus."
     }
   }
   ```
   
   > **Important**: You must provide translations for **all** message keys. Missing keys will cause the bot to display the key name instead of the translated text. Check `breeds_ru.json` and the existing language sections in `messages.json` to ensure completeness.

3. **Set the language**: Add `LANGUAGE=<lang>` to your `.env` file (e.g., `LANGUAGE=de`).

4. **Restart the bot**: The bot will automatically detect and use the new language files on startup. If files are missing, a warning will be printed and the bot will fall back to Russian (`ru`).

> **Note**: 
> - Use short language codes (2-3 letters) following common conventions (e.g., `de`, `fr`, `es`, `it`).
> - The bot automatically validates translation files on startup. If either `breeds_<lang>.json` or the language section in `messages.json` is missing, the bot will print a warning and use Russian as a fallback.