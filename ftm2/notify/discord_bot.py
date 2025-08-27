import asyncio
import discord

from ftm2.config.settings import load_env_chain

CFG = load_env_chain()


class DiscordBot(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        super().__init__(intents=intents)
        self.signals_chan = None
        self.trades_chan = None
        self.logs_chan = None

    async def on_ready(self):
        self.signals_chan = self.get_channel(CFG.DISCORD_CHANNEL_SIGNALS)
        self.trades_chan = self.get_channel(CFG.DISCORD_CHANNEL_TRADES)
        self.logs_chan = self.get_channel(CFG.DISCORD_CHANNEL_LOGS)
        print(f"Logged in as {self.user}")

    async def send_signal(self, content: str):
        if self.signals_chan:
            await self.signals_chan.send(content)


    async def send_log(self, content: str):
        if self.logs_chan:
            await self.logs_chan.send(content)



async def run_bot():
    bot = DiscordBot()
    await bot.start(CFG.DISCORD_TOKEN)
