import requests
import re
from bs4 import BeautifulSoup

cleaner = re.compile('<.*?>')
result = []
url = "https://nitter.net/search?f=tweets&q=%22Anies+Baswedan%22+Presiden&since=2023-03-01&until=2023-03-31"
def get_tweet(link):
    r = requests.get(url=link)

    html = r.content
    soup = BeautifulSoup(html, 'lxml')

    response = soup.find_all("div", class_="tweet-content media-body")
    for res in response:
        result.append(re.sub(cleaner, '', str(res)))

get_tweet(url)
print(result)