from discord_webhook import DiscordWebhook, DiscordEmbed
import time

webhook_url = 'https://discord.com/api/webhooks/1120721392388800582/wF8-SK_jrV4djlpYq4p6NREqGejLoD2NKwh9n--YhdntsxA91bbHGmmzCrU-lUYQs3hW'
global webhook
webhook = DiscordWebhook(url=webhook_url, rate_limit_retry=True)

start = time.time()

def uptime():
    return time.strftime("%H:%M:%S", time.gmtime(end - start))


def initial_message():
    # create embed object for webhook
    embed = DiscordEmbed(
        title="Embed Title", description="Your Embed Description", color='03b2f8'
    )
    # set author
    embed.set_author(name='Author Name')
    # set timestamp (default is now)
    embed.set_timestamp()
    # add fields to embed
    embed.add_embed_field(name='Field 1', value='Lorem ipsum')
    embed.add_embed_field(name='Field 2', value='dolor sit')
    # add embed object to webhook
    webhook.add_embed(embed)
    webhook.execute()



def modify_message(webh,symbol, timeframe, estimed_time_realizing, minutes,predicted_price=float, actual_price=float):
    difference_percentage = (float(predicted_price) - float(actual_price)) / float(actual_price) * 100
    webh.content = 'AI BOT - Version 5'
    webh.username = 'V5'
    webh.avatar_url = 'https://i.imgur.com/rdm3W9t.png'
    
    # set color to red or green
    webh.files = []
    webh.embeds = []
    
    # create embed object for webhook
    # embed = DiscordEmbed(title='Your Title', description='Lorem ipsum dolor sit', color='03b2f8')
    embed = DiscordEmbed(
        title="Embed Title", description="Your Embed Description", color='03b2f8'
    )
    
    # set the fields with arguments
    embed.add_embed_field(name='Symbol', value="```" + str(symbol) + " " + str(timeframe) + "```")
    embed.add_embed_field(name='Actual Price', value="```" + str(actual_price)+ "```")
    embed.add_embed_field(name='Predicted Price', value="```" + str(predicted_price) + "```", inline=False)
    embed.add_embed_field(name='Estimed Time Realizing', value="```" + str(estimed_time_realizing)+ "```")
    embed.add_embed_field(name='Minutes', value="```" + str(minutes)+ "```")
    embed.add_embed_field(name='Difference Percentage', value="```" + str(difference_percentage)+ "```")
    
    # with open("images/image.jpg", "rb") as f:
        # webhook.add_file(file=f.read(), filename='image.jpg')
    webh.add_embed(embed)
    webh.edit()



initial_message()
modify_message(webhook,symbol="BTCUSDT", predicted_price=10000, actual_price=10000, timeframe="1m", estimed_time_realizing="1m", minutes="1m")