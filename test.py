from discord_webhook import DiscordWebhook, DiscordEmbed

from time import sleep

webhook_url = 'https://discord.com/api/webhooks/1120721392388800582/wF8-SK_jrV4djlpYq4p6NREqGejLoD2NKwh9n--YhdntsxA91bbHGmmzCrU-lUYQs3hW'
global webhook
webhook = DiscordWebhook(url=webhook_url, rate_limit_retry=True)
# embed = DiscordEmbed(
#     title="Embed Title", description="Your Embed Description", color='03b2f8'
# )
# embed.set_author(
#     name="Author Name",
#     url="https://github.com/lovvskillz",
#     icon_url="https://avatars0.githubusercontent.com/u/14542790",
# )
# embed.set_footer(text="Embed Footer Text")
# embed.set_timestamp()

# # Set `inline=False` for the embed field to occupy the whole line
# embed.add_embed_field(name="Field 1", value="Lorem ipsum", inline=False)
# embed.add_embed_field(name="Field 2", value="dolor sit", inline=False)
# embed.add_embed_field(name="Field 3", value="amet consetetur")
# embed.add_embed_field(name="Field 4", value="sadipscing elitr")

# webhook.add_embed(embed)
# response = webhook.execute()






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



def modify_message(symbol, predicted_price, actual_price, timeframe, estimed_time_realizing, minutes, difference_percentage):
    difference_percentage
    webhook.embeds = []
    # create embed object for webhook
    # embed = DiscordEmbed(title='Your Title', description='Lorem ipsum dolor sit', color='03b2f8')
    embed = DiscordEmbed(
        title="Embed Title", description="Your Embed Description", color='03b2f8'
    )
    # set the fields with arguments
    embed.add_embed_field(name='Symbol', value="```" + str(symbol) + " " + str(timeframe) + "```")
    embed.add_embed_field(name='Predicted Price', value="```" + str(predicted_price) + "```")
    embed.add_embed_field(name='Actual Price', value="```" + actual_price+ "```")
    embed.add_embed_field(name='Estimed Time Realizing', value="```" + estimed_time_realizing+ "```")
    embed.add_embed_field(name='Minutes', value="```" + minutes+ "```")
    embed.add_embed_field(name='Difference Percentage', value="```" + difference_percentage+ "```")
    # with open("images/image.jpg", "rb") as f:
        # webhook.add_file(file=f.read(), filename='image.jpg')
    webhook.add_embed(embed)
    webhook.edit()



initial_message()
modify_message(symbol="BTCUSDT", predicted_price="10000", actual_price="10000", timeframe="1m", estimed_time_realizing="1m", minutes="1m", difference_percentage="1m")