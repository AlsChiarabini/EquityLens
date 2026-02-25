from equitylens.data.fetcher import DataFetcher
fetcher = DataFetcher()
data = fetcher.fetch("AAPL")
print(f"STAMPO I DATI {data.ticker}\n")
print(data.profile)
print("------------------\n")
print(f"STAMPO I PREZZI DI {data.ticker}")
print("------------------\n")
print(data.prices.tail())
print(f"News: {len(data.news)} articles")
print(f"Transcripts: {len(data.transcripts)}")
for t in data.transcripts:
    print(f"  - {t['title']}  ({len(t['content'])} chars)")
print(f"Peers: {data.peers}")
