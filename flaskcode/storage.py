from contract import *
import os
import pickle
import json
from web3 import Web3
import time

ganache_url = "http://127.0.0.1:8545"
web3 = Web3(Web3.HTTPProvider(ganache_url, request_kwargs={'timeout': 1000000000}))
a,tx_hash = deploy(web3)
# print(web3)
print(tx_hash)
# tx_hash='0xe78A0F7E598Cc8b0Bb87894B0F60dD2a88d6a8Ab'
with open('txhash.txt','w') as f:
    f.write(tx_hash)

storage(web3,tx_hash)
