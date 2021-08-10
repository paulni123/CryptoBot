from Constants import TestingMonths
from Account import Account
import numpy as np


class CryptoTrader:

    def __init__(self, model):
        self.model = model
        self.account = Account()
        self.tradeAmount = 100

    
    def buy(self):
        prevBought = self.account.bought_BTC_At
        if self.account.USD_Balance - self.tradeAmount >= 0:
            if prevBought == 0 or self.account.prevWasSell or (prevBought > self.account.BTC_Price):
                print("Buying $", self.tradeAmount, " worth of BTC")
                self.account.BTC_Amount += (self.tradeAmount / self.account.BTC_Price) #adding the amount of BTC bought
                self.account.USD_Balance -= self.tradeAmount 
                self.account.bought_BTC_At = self.account.BTC_Price
                self.account.prevWasSell = False
            else:
                print("Doesn't make sense to buy more BTC at this moment")
        else:
            print("Not enough USD in account to buy more BTC")

    
    def sell(self):
        if self.account.BTC_Balance - self.tradeAmount >= 0:
            if self.account.BTC_Price > self.account.bought_BTC_At:
                print("Selling $", self.tradeAmount, " worth of BTC")
                self.account.BTC_Amount -= (self.tradeAmount / self.account.BTC_Price)
                self.account.USD_Balance += self.tradeAmount
                self.account.prevWasSell = True
            else:
                print("Not profitable to be selling BTC at this moment")
        else:
            print("Not enough BTC left in account to sell, need at least ",
            (self.tradeAmount/self.account.BTC_Price), " of BTC or $", 
            self.tradeAmount, " worth of BTC")



    def simulate(self,samples,prices):
        print("Simulating the automatic trading for ", TestingMonths)
        dayCount = 0
        for i in range(0, len(samples)):

            if i % 288 == 0:
                dayCount += 1
                print("Account Balance: $", (self.account.USD_Balance + self.account.BTC_Balance), "\nBTC: $",
                      self.account.BTC_Balance, "\nUSD: $", self.account.USD_Balance, "")
                print("Day: " + str(dayCount))

            if i % 3 == 0:
                prediction = self.model.predict(np.array([samples[i]]))
                btcPrice = prices[i]

                self.account.BTC_Price = btcPrice

                if prediction == 1:
                    self.buy()
                elif prediction == 0.5:
                    self.sell()
                
                self.account.BTC_Balance = self.account.BTC_Amount * btcPrice


        print("Account Balance: $", (self.account.USD_Balance + self.account.BTC_Balance), "\nBTC: $",
        self.account.BTC_Balance, "\nUSD: $", self.account.USD_Balance, "")