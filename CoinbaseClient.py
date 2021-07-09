import cbpro
import os
from dotenv import load_dotenv

class CoinbaseClient:
    
    public_Client = None
    auth_Client = None

    def __init__(self):
       load_dotenv()
       self.public_Client = cbpro.PublicClient()
       self.auth_Client = cbpro.AuthenticatedClient(os.getenv('API_KEY'),os.getenv('API_SECRET'),os.getenv('PASSPHRASE'))

    def test(self):
        print(os.getenv('API_KEY'))
        print(self.public_Client.get_time())
        print(self.auth_Client.get_accounts())
        print()
        print(self.public_Client.get_product_historic_rates('BTC-USD'))


    def getCoinHistoricalData(self, tickerId, start, end, granularity):
        return self.public_Client.get_product_historic_rates(tickerId,start,end,granularity)


client = CoinbaseClient()
#client.test()