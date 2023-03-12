import logging
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import dotenv_values
from langchain import LLMChain, PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.llms import OpenAIChat
from pyrogram import filters
from pyrogram.client import Client
from pyrogram.types import Message
import openai
from pydub import AudioSegment

TZ = ZoneInfo("Asia/Singapore")

os.environ["OPENAI_API_KEY"] = dotenv_values(Path(__file__).parent / ".env")[
    "OPENAI_API_KEY"
]

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)

app = Client("my_account")

# Initialize LLM
llm = OpenAIChat(temperature=0.2)
memory = ConversationBufferWindowMemory(k=5)
template = (
    "You are an AI. Answer as comprehensively as possible. The current location is Singapore and the date and time is "
    + datetime.now(tz=ZoneInfo("Asia/Singapore")).strftime("%c %z")
    + ". \n\n{history}\n\n Human: {human_input}\nAI:"
)

prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)

chatgpt_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
)


@app.on_message(filters.command("clear"))
async def handle_clear(client, message: Message):
    memory.clear()
    logger.info("Memory cleared")
    await message.reply("Memory cleared")


@app.on_message(filters.text)
async def handle_text(client, message: Message):
    if message.from_user.is_self:
        return
    logger.info(f"Message received: {message.text}")
    response = chatgpt_chain.predict(human_input=message.text)
    await message.reply(response, quote=True)

@app.on_message(filters.voice)
async def handle_voice(client, message: Message):
    path = await message.download()
    audio = AudioSegment.from_ogg(path)
    audio.export(f"{path}.mp3", format='mp3')
    with open(f"{path}.mp3", 'rb') as f:
        text = openai.Audio.transcribe('whisper-1', f, language='en').get('text', '')
        await message.reply(f"You asked: {text}")
        response = chatgpt_chain.predict(human_input=text)
        await message.reply(response, quote=True)

app.run()