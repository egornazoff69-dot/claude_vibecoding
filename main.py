import json
import logging
import os
from pathlib import Path
from collections import defaultdict
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import AsyncOpenAI
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config & constants
# ---------------------------------------------------------------------------

LLM_BASE_URL = os.environ["LLM_BASE_URL"]
LLM_API_KEY = os.environ["LLM_API_KEY"]
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
CHAT2DESK_TOKEN = os.environ["CHAT2DESK_TOKEN"]
CHAT2DESK_API = "https://api.chat2desk.com/v1/messages"
YCLIENTS_API = "https://api.yclients.com/api/v1"
CONFIGS_DIR = Path("configs")
MAX_HISTORY = 20

llm_client = AsyncOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

# chat_id -> list of message dicts (role/content)
conversations: dict[str, list[dict]] = defaultdict(list)

app = FastAPI(title="LLM Webhook Server")

# ---------------------------------------------------------------------------
# Tools schema passed to LLM
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_available_slots",
            "description": (
                "Get available booking time slots at the salon for a specific date. "
                "Returns a list of available datetime strings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format",
                    }
                },
                "required": ["date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_booking",
            "description": "Create a booking for a client at the salon.",
            "parameters": {
                "type": "object",
                "properties": {
                    "datetime": {
                        "type": "string",
                        "description": "Booking datetime in ISO 8601 format, e.g. 2025-06-15T14:00:00",
                    },
                    "client_name": {
                        "type": "string",
                        "description": "Full name of the client",
                    },
                    "client_phone": {
                        "type": "string",
                        "description": "Client phone number in international format, e.g. +79001234567",
                    },
                    "service_id": {
                        "type": "integer",
                        "description": "Yclients service ID to book",
                    },
                },
                "required": ["datetime", "client_name", "client_phone", "service_id"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_config(channel_id: str) -> dict:
    """Load channel config from configs/{channel_id}.json."""
    config_path = CONFIGS_DIR / f"{channel_id}.json"
    if not config_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Config not found for channel_id='{channel_id}'. "
                   f"Expected file: {config_path}",
        )
    with config_path.open() as f:
        return json.load(f)


def trim_history(chat_id: str) -> None:
    """Keep only the last MAX_HISTORY messages for this chat."""
    history = conversations[chat_id]
    if len(history) > MAX_HISTORY:
        conversations[chat_id] = history[-MAX_HISTORY:]


# ---------------------------------------------------------------------------
# Yclients API calls
# ---------------------------------------------------------------------------


async def get_available_slots(date: str, config: dict) -> dict:
    """Fetch available booking slots from Yclients for a given date."""
    company_id = config["yclients_company_id"]
    staff_id = config["staff_id"]
    service_id = config.get("service_id", "")
    api_key = config["yclients_api_key"]

    url = f"{YCLIENTS_API}/book_times/{company_id}/{staff_id}/{service_id}/{date}"
    headers = {
        "Authorization": f"Bearer {api_key}, User {api_key}",
        "Accept": "application/vnd.yclients.v2+json",
    }
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    slots = [item.get("time") or item.get("datetime") for item in data.get("data", [])]
    logger.info("Available slots for %s: %s", date, slots)
    return {"date": date, "available_slots": slots}


async def create_booking(
    booking_datetime: str,
    client_name: str,
    client_phone: str,
    service_id: int,
    config: dict,
) -> dict:
    """Create a booking via Yclients API."""
    company_id = config["yclients_company_id"]
    staff_id = config["staff_id"]
    api_key = config["yclients_api_key"]

    url = f"{YCLIENTS_API}/bookrecord/{company_id}"
    headers = {
        "Authorization": f"Bearer {api_key}, User {api_key}",
        "Accept": "application/vnd.yclients.v2+json",
        "Content-Type": "application/json",
    }
    # Split name into first/last (Yclients requires separate fields)
    name_parts = client_name.strip().split(None, 1)
    first_name = name_parts[0]
    last_name = name_parts[1] if len(name_parts) > 1 else ""

    payload = {
        "phone": client_phone,
        "fullname": client_name,
        "name": first_name,
        "surname": last_name,
        "appointments": [
            {
                "id": 1,
                "services": [service_id],
                "staff_id": staff_id,
                "datetime": booking_datetime,
            }
        ],
    }
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    logger.info("Booking created: %s", data)
    return {"success": True, "booking": data.get("data", data)}


# ---------------------------------------------------------------------------
# Chat2Desk
# ---------------------------------------------------------------------------


async def send_chat2desk(chat_id: str, text: str) -> None:
    """Send a reply message back to the client via Chat2Desk API."""
    headers = {
        "Authorization": CHAT2DESK_TOKEN,
        "Content-Type": "application/json",
    }
    payload = {"chat_id": chat_id, "text": text, "type": "text"}
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(CHAT2DESK_API, headers=headers, json=payload)
        resp.raise_for_status()
    logger.info("Sent reply to chat_id=%s via Chat2Desk", chat_id)


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------


async def dispatch_tool(tool_name: str, args: dict, config: dict) -> Any:
    """Call the appropriate Yclients function and return a JSON-serialisable result."""
    if tool_name == "get_available_slots":
        return await get_available_slots(date=args["date"], config=config)

    if tool_name == "create_booking":
        return await create_booking(
            booking_datetime=args["datetime"],
            client_name=args["client_name"],
            client_phone=args["client_phone"],
            service_id=args["service_id"],
            config=config,
        )

    return {"error": f"Unknown tool: {tool_name}"}


# ---------------------------------------------------------------------------
# LLM interaction with tool-call loop
# ---------------------------------------------------------------------------


async def run_llm(chat_id: str, config: dict) -> str:
    """
    Send the conversation to the LLM, handle any tool calls, and return
    the final text response.
    """
    system_prompt = config["system_prompt"]

    while True:
        messages = [{"role": "system", "content": system_prompt}] + conversations[chat_id]

        response = await llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        choice = response.choices[0]
        assistant_message = choice.message

        # If the LLM wants to call tools, execute them and loop
        if assistant_message.tool_calls:
            # Append the assistant turn (with tool_calls) to history
            conversations[chat_id].append(assistant_message.model_dump(exclude_unset=True))

            for tc in assistant_message.tool_calls:
                tool_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                    logger.info("Tool call: %s(%s)", tool_name, args)
                    result = await dispatch_tool(tool_name, args, config)
                    content = json.dumps(result, ensure_ascii=False)
                except Exception as exc:
                    logger.exception("Tool %s failed", tool_name)
                    content = json.dumps({"error": str(exc)})

                conversations[chat_id].append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": content,
                    }
                )

            trim_history(chat_id)
            # Loop: send tool results back to LLM
            continue

        # No tool calls — we have the final text response
        reply = assistant_message.content or ""
        conversations[chat_id].append({"role": "assistant", "content": reply})
        trim_history(chat_id)
        return reply


# ---------------------------------------------------------------------------
# Webhook endpoint
# ---------------------------------------------------------------------------


class WebhookPayload(BaseModel):
    chat_id: str
    message: str
    client_id: str
    channel_id: str


@app.post("/webhook")
async def webhook(payload: WebhookPayload) -> dict:
    logger.info(
        "Webhook received: chat_id=%s channel_id=%s client_id=%s",
        payload.chat_id,
        payload.channel_id,
        payload.client_id,
    )

    # 1. Load channel config (raises 404 if missing)
    config = load_config(payload.channel_id)

    # 2. Append user message to history (trim first so new message is always kept)
    trim_history(payload.chat_id)
    conversations[payload.chat_id].append(
        {"role": "user", "content": payload.message}
    )

    # 3. Run LLM (with tool-call loop)
    try:
        reply = await run_llm(payload.chat_id, config)
    except Exception as exc:
        logger.exception("LLM error for chat_id=%s", payload.chat_id)
        raise HTTPException(status_code=502, detail=f"LLM error: {exc}") from exc

    # 4. Send reply to client via Chat2Desk
    try:
        await send_chat2desk(payload.chat_id, reply)
    except Exception as exc:
        logger.exception("Chat2Desk send error for chat_id=%s", payload.chat_id)
        raise HTTPException(status_code=502, detail=f"Chat2Desk error: {exc}") from exc

    return {"status": "ok", "reply": reply}


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
