import discord_webhook
import dotenv
import os
import sys

dotenv.load_dotenv()

class Notifier:
    users = {
        "idev": 707901520217374811
    }
    def __init__(self, **kwargs):
        url = os.getenv('DISCORD')
        if url is not None:
            self._discord = discord_webhook.DiscordWebhook(url, kwargs=kwargs)

    def _content(self, c):
        self._discord.set_content(c[-2000:])

    def notify(self, message, mentions=[]):
        print(mentions)
        if hasattr(self, '_discord'):
            s = [self.users[user] for user in mentions if user in self.users]
            m = " ".join([f'<@{u}>' for u in s])
            self._content(f'{m} {message}')
            self._discord.execute()
    
    def upload(self, files=[], message=""):
        if hasattr(self, '_discord'):
            for file in files:
                with open(file, 'rb') as f:
                   self._discord.add_file(file=f.read(), filename=file.split('/')[-1])
            self._content(message)
            self._discord.execute()
    
if __name__ == '__main__':
    notifier = Notifier()
    notifier.notify(sys.argv[1], sys.argv[2:])
