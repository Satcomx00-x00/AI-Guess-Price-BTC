import discord
from discord.ext import commands
import datetime
import os, json
import asyncio
# RuntimeWarning: Enable tracemalloc to get the object allocation traceback
import tracemalloc

tracemalloc.start()
# r'C:\Users\MrBios\Documents\Development\test\production\Docker\csv\prediction.csv'

# from dotenv import load_dotenv

# load_dotenv()

# TOKEN = os.getenv('DISCORD_TOKEN')
# CHANNEL = os.getenv('DISCORD_CHANNEL_ID')

# {"ticker": "BTC/USD", "prochain_prix": 29247.90234375, "last_price": 28208.5, "percent": 0.03684713273481397, "uncertainty": 0.0, "eta": "1.074975"}
# make a dict of key with a name
names = {
    'ticker': 'Ticker',
    'prochain_prix': 'Next Price',
    'last_price': 'Last Price',
    'percent': 'Percent',
    'uncertainty': 'Uncertainty',
    'eta': 'ETA'
}


class PredictionMessage:

    def __init__(self, token, channel_id):
        self.token = str(token)
        self.channel_id = int(channel_id)
        self.bot = None
        self.message = None

    async def send_embed_message(self, msg):
        channel = self.bot.get_channel(self.channel_id)
        self.message = await channel.send(embed=msg)
        return self.message

    async def modify_message(self, new_content):
        await self.message.edit(content=new_content)

    async def modify_embed_message(self, embed):
        if self.message is not None:
            await self.message.edit(embed=embed)

    async def on_ready(self):
        print(f'{self.bot.user} has connected to Discord!')
        # if message_id.txt exists, read the message ID from it
        if os.path.exists('message_id.txt'):
            with open('message_id.txt', 'r') as f:
                print('Reading message ID from file...')
                message_id = int(f.read())
                self.message = await self.bot.get_channel(
                    self.channel_id).fetch_message(message_id)

        # Send the initial message
        else:
            print('Sending initial message...')
            with open('json/prediction.json', 'r') as f:
                data = json.load(f)
            print(json.dumps(data, indent=4))
            embed = discord.Embed(title='Prediction',
                                  description='New prediction available',
                                  color=0x00ff00)

            for key, value in data.items():
                embed.add_field(name=key, value=value, inline=True)
            print(embed)
            self.message = await self.send_embed_message(embed)
            print(self.message)

            # Store the message ID for future modification
            with open('message_id.txt', 'w') as f:
                f.write(str(self.message.id))

    async def update_prediction(self):
        print('Updating prediction...')
        with open('json/prediction.json', 'r') as f:
            data = json.load(f)
        embed = discord.Embed(title='Prediction update',
                              color=discord.Color.blue())
        for key, value in data.items():
            # if it floats, round it to 2 decimal places
            if isinstance(value, float):
                value = round(value, 3)
            if key == 'eta':
                # value = round(float(value) / 60, 3)
                # calcul in hours
                # value = round(float(value) / 3600, 3)
                # 18.81 hours in hh:mm:ss
                
                value = float(value)
                value = str(datetime.timedelta(seconds=value * 3600))

                
            # same for percentage
            if key == 'percent':
                # if prediction is positive, add a plus sign
                if float(data["prochain_prix"]) > float(data["last_price"]):
                    value = '+' + str(round(float(value) * 100, 3)) + '%'
                else:
                    value = str(round(float(value) * 100, 3)) + '%'
            embed.add_field(name=names[key],
                            value="```" + str(value) + "```",
                            inline=True)
        embed.timestamp = datetime.datetime.utcnow() + datetime.timedelta(
            hours=2)
        await self.modify_embed_message(embed=embed)

    async def on_message(self, message):
        # Ignore messages sent by the bot itself
        if message.author == self.bot.user:
            return

    async def start_timer(self):
        # print the current time every 3 seconds
        await self.bot.wait_until_ready()
        await asyncio.sleep(4)

        try:
            while not self.bot.is_closed():
                await self.update_prediction()
                await asyncio.sleep(30)
        except Exception as e:
            # send the error in the channel
            print(e)
            channel = self.bot.get_channel(self.channel_id)
            self.message = await channel.send(e)

    async def run(self):
        print('Starting bot...')
        intents = discord.Intents.default()
        intents.members = True
        intents.presences = True
        intents.messages = True
        intents.message_content = True

        self.bot = commands.Bot(command_prefix='?', intents=intents)

        @self.bot.event
        async def on_ready():
            await self.on_ready()

        @self.bot.event
        async def on_message(message):
            await self.on_message(message)

        asyncio.create_task(self.start_timer())
        # task = self.bot.loop.create_task(self.start_timer())
        await self.bot.start(self.token)


# # if __name__ == '__main__':
# prediction_message = PredictionMessage(TOKEN, CHANNEL)
# asyncio.run(prediction_message.run())
