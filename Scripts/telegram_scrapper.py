
from telethon import TelegramClient
from dotenv import load_dotenv
import csv
import os
class Scrapper:
    def __init__(self):
        self.df = {}
    def DataCrawller(self):
        # channels = ['@ZemenExpress','@nevacomputer','@meneshayeofficial','@ethio_brand_collection','@Leyueqa', '@sinayelj',
        #  '@Shewabrand','@helloomarketethiopia','@modernshoppingcenter','@qnashcom','@Fashiontera','@kuruwear',
        #  '@gebeyaadama','@MerttEka','@forfreemarket','@classybrands','@marakibrand','@aradabrand2','@marakisat2',
        # '@belaclassic','@AwasMart','@qnashcom']

        # Load environment variables
        load_dotenv('.env')
        self.api_id = os.getenv('TG_API_ID')
        self.api_hash = os.getenv('TG_API_HASH')
        self.phone = os.getenv('phone')

        # Function to scrape data from a single channel
        async def scrape_channel(client, channel_username, writer, media_dir):
            entity = await client.get_entity(channel_username)
            channel_title = entity.title  # Extract the channel's title
            async for message in client.iter_messages(entity, limit=10000):
                media_path = None
                if message.media and hasattr(message.media, 'photo'):
                    # Create a unique filename for the photo
                    filename = f"{channel_username}_{message.id}.jpg"
                    media_path = os.path.join(media_dir, filename)
                    # Download the media to the specified directory if it's a photo
                    await client.download_media(message.media, media_path)

                # Write the channel title along with other data
                writer.writerow([channel_title, channel_username, message.id, message.message, message.date, media_path])

        # Initialize the client once
        client = TelegramClient('scraping_session', api_id, api_hash)

        async def main():
            await client.start()

            # Create a directory for media files
            media_dir = 'photos2'
            os.makedirs(media_dir, exist_ok=True)

            # Open the CSV file and prepare the writer
            # with open('file.txt', 'w', encoding='utf-8') as file:
            #   file.write(amharic_text)
            with open('telegram_data.txt', 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['Channel Title', 'Channel Username', 'ID', 'Message', 'Date', 'Media Path'])  # Include channel title in the header

                # List of channels to scrape
                channels = [
                    '@Shageronlinestore',  # Existing channel
                    # You can add more channels here
                ]

                # Iterate over channels and scrape data into the single CSV file
                for channel in channels:
                    await scrape_channel(client, channel, writer, media_dir)
                    print(f"Scraped data from {channel}")

        # Use async with for the client
        with client:
             main()
        # async with client:
        #     await main()
