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
    "The current location is Singapore and the date and time is "
    + datetime.now(tz=ZoneInfo("Asia/Singapore")).strftime("%c %z")
    + ". Answer as concisely as possible.\n\n{history}\n\n Human: {human_input}\nAI:"
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
    logger.info(f"Message received: {message.text}")
    response = chatgpt_chain.predict(human_input=message.text)
    await message.reply(response, quote=True)


app.run()
