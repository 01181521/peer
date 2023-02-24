import os
import pickle
import json
from web3 import Web3
import time


f1 = open(r"./model/database.txt","r")
file1 = f1.read()



abi = json.loads('''[
	{
		"constant": true,
		"inputs": [],
		"name": "get",
		"outputs": [
			{
				"internalType": "uint256",
				"name": "",
				"type": "uint256"
			}
		],
		"payable": false,
		"stateMutability": "view",
		"type": "function"
	},
	{
		"constant": true,
		"inputs": [
			{
				"internalType": "uint256",
				"name": "index",
				"type": "uint256"
			}
		],
		"name": "getElement",
		"outputs": [
			{
				"internalType": "bytes1",
				"name": "",
				"type": "bytes1"
			}
		],
		"payable": false,
		"stateMutability": "view",
		"type": "function"
	},
	{
		"constant": true,
		"inputs": [],
		"name": "getString",
		"outputs": [
			{
				"internalType": "string",
				"name": "",
				"type": "string"
			}
		],
		"payable": false,
		"stateMutability": "view",
		"type": "function"
	},
	{
		"constant": true,
		"inputs": [],
		"name": "getindex",
		"outputs": [
			{
				"internalType": "uint256[]",
				"name": "",
				"type": "uint256[]"
			},
			{
				"internalType": "uint256[]",
				"name": "",
				"type": "uint256[]"
			}
		],
		"payable": false,
		"stateMutability": "view",
		"type": "function"
	},
	{
		"constant": true,
		"inputs": [
			{
				"internalType": "uint256",
				"name": "index",
				"type": "uint256"
			}
		],
		"name": "getint",
		"outputs": [
			{
				"internalType": "uint256",
				"name": "",
				"type": "uint256"
			}
		],
		"payable": false,
		"stateMutability": "view",
		"type": "function"
	},
	{
		"constant": false,
		"inputs": [
			{
				"internalType": "bytes",
				"name": "value",
				"type": "bytes"
			}
		],
		"name": "push",
		"outputs": [],
		"payable": false,
		"stateMutability": "nonpayable",
		"type": "function"
	},
	{
		"constant": false,
		"inputs": [
			{
				"internalType": "bytes",
				"name": "value",
				"type": "bytes"
			}
		],
		"name": "search",
		"outputs": [
			{
				"internalType": "uint256[]",
				"name": "",
				"type": "uint256[]"
			},
			{
				"internalType": "uint256[]",
				"name": "",
				"type": "uint256[]"
			}
		],
		"payable": false,
		"stateMutability": "nonpayable",
		"type": "function"
	},
	{
		"constant": false,
		"inputs": [
			{
				"internalType": "bytes",
				"name": "value",
				"type": "bytes"
			},
			{
				"internalType": "uint256",
				"name": "index",
				"type": "uint256"
			}
		],
		"name": "topk",
		"outputs": [
			{
				"internalType": "uint256[]",
				"name": "",
				"type": "uint256[]"
			},
			{
				"internalType": "uint256[]",
				"name": "",
				"type": "uint256[]"
			}
		],
		"payable": false,
		"stateMutability": "nonpayable",
		"type": "function"
	}
]''')

def deploy(web3):
	# ganache_url = "http://127.0.0.1:8545"
	# web3 = Web3(Web3.HTTPProvider(ganache_url, request_kwargs={'timeout': 10000000}))
	
	# web3 = Web3(Web3.IPCProvider())
	print(web3.isConnected())

	bytecode = "60806040526040518060400160405280600581526020017f68656c6c6f000000000000000000000000000000000000000000000000000000815250600190805190602001906200005192919062000145565b506040518060400160405280600481526020017f7472756500000000000000000000000000000000000000000000000000000000815250600290805190602001906200009f92919062000145565b506040518060400160405280600381526020017f796573000000000000000000000000000000000000000000000000000000000081525060039080519060200190620000ed92919062000145565b50602360f81b600460006101000a81548160ff021916908360f81c021790555060b260f81b600460016101000a81548160ff021916908360f81c021790555060006007553480156200013e57600080fd5b50620001f4565b828054600181600116156101000203166002900490600052602060002090601f016020900481019282601f106200018857805160ff1916838001178555620001b9565b82800160010185558215620001b9579182015b82811115620001b85782518255916020019190600101906200019b565b5b509050620001c89190620001cc565b5090565b620001f191905b80821115620001ed576000816000905550600101620001d3565b5090565b90565b6111c580620002046000396000f3fe608060405234801561001057600080fd5b50600436106100885760003560e01c8063879647e21161005b578063879647e21461022e57806389ea642f146102d5578063bc8ee6fc14610358578063ffe47662146104ba57610088565b806324d5ab441461008d5780633a7d22bc146100cf5780636d4ce63c146101555780637dacda0314610173575b600080fd5b6100b9600480360360208110156100a357600080fd5b8101908080359060200190929190505050610612565b6040518082815260200191505060405180910390f35b6100fb600480360360208110156100e557600080fd5b8101908080359060200190929190505050610633565b60405180827effffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff19167effffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff1916815260200191505060405180910390f35b61015d61069e565b6040518082815260200191505060405180910390f35b61022c6004803603602081101561018957600080fd5b81019080803590602001906401000000008111156101a657600080fd5b8201836020820111156101b857600080fd5b803590602001918460018302840111640100000000831117156101da57600080fd5b91908080601f016020809104026020016040519081016040528093929190818152602001838380828437600081840152601f19601f8201169050808301925050505050505091929192905050506106a8565b005b610236610793565b604051808060200180602001838103835285818151815260200191508051906020019060200280838360005b8381101561027d578082015181840152602081019050610262565b50505050905001838103825284818151815260200191508051906020019060200280838360005b838110156102bf5780820151818401526020810190506102a4565b5050505090500194505050505060405180910390f35b6102dd610846565b6040518080602001828103825283818151815260200191508051906020019080838360005b8381101561031d578082015181840152602081019050610302565b50505050905090810190601f16801561034a5780820380516001836020036101000a031916815260200191505b509250505060405180910390f35b61041b6004803603604081101561036e57600080fd5b810190808035906020019064010000000081111561038b57600080fd5b82018360208201111561039d57600080fd5b803590602001918460018302840111640100000000831117156103bf57600080fd5b91908080601f016020809104026020016040519081016040528093929190818152602001838380828437600081840152601f19601f82011690508083019250505050505050919291929080359060200190929190505050610a49565b604051808060200180602001838103835285818151815260200191508051906020019060200280838360005b83811015610462578082015181840152602081019050610447565b50505050905001838103825284818151815260200191508051906020019060200280838360005b838110156104a4578082015181840152602081019050610489565b5050505090500194505050505060405180910390f35b610573600480360360208110156104d057600080fd5b81019080803590602001906401000000008111156104ed57600080fd5b8201836020820111156104ff57600080fd5b8035906020019184600183028401116401000000008311171561052157600080fd5b91908080601f016020809104026020016040519081016040528093929190818152602001838380828437600081840152601f19601f820116905080830192505050505050509192919290505050610dcc565b604051808060200180602001838103835285818151815260200191508051906020019060200280838360005b838110156105ba57808201518184015260208101905061059f565b50505050905001838103825284818151815260200191508051906020019060200280838360005b838110156105fc5780820151818401526020810190506105e1565b5050505090500194505050505060405180910390f35b60006005828154811061062157fe5b90600052602060002001549050919050565b6000808281546001816001161561010002031660029004811061065257fe5b8154600116156106715790600052602060002090602091828204019190065b9054901a7f0100000000000000000000000000000000000000000000000000000000000000029050919050565b6000600754905090565b60008090505b815181101561078f5760008282815181106106c557fe5b602001015160f81c60f81b90808054603f811680603e811461070157600283018455600183166106f3578192505b60016002840401935061071b565b83600052602060002060ff19841681556041855560209450505b50505090600182038154600116156107425790600052602060002090602091828204019190065b90919290919091601f036101000a81548160ff021916907f0100000000000000000000000000000000000000000000000000000000000000840402179055505080806001019150506106ae565b5050565b60608060056006818054806020026020016040519081016040528092919081815260200182805480156107e557602002820191906000526020600020905b8154815260200190600101908083116107d1575b505050505091508080548060200260200160405190810160405280929190818152602001828054801561083757602002820191906000526020600020905b815481526020019060010190808311610823575b50505050509050915091509091565b6060600460009054906101000a900460f81b7effffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff19166000600181546001816001161561010002031660029004811061089957fe5b8154600116156108b85790600052602060002090602091828204019190065b9054901a7f0100000000000000000000000000000000000000000000000000000000000000027effffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff191614156109a85760028054600181600116156101000203166002900480601f01602080910402602001604051908101604052809291908181526020018280546001816001161561010002031660029004801561099c5780601f106109715761010080835404028352916020019161099c565b820191906000526020600020905b81548152906001019060200180831161097f57829003601f168201915b50505050509050610a46565b60018054600181600116156101000203166002900480601f016020809104026020016040519081016040528092919081815260200182805460018160011615610100020316600290048015610a3e5780601f10610a1357610100808354040283529160200191610a3e565b820191906000526020600020905b815481529060010190602001808311610a2157829003601f168201915b505050505090505b90565b60608060008090506000809050600080905060006007819055506000600581610a72919061113f565b506000600681610a82919061113f565b5060008090505b6002600080546001816001161561010002031660029004905081610aa957fe5b04811015610d1457866007541415610b70576005600681805480602002602001604051908101604052809291908181526020018280548015610b0a57602002820191906000526020600020905b815481526020019060010190808311610af6575b5050505050915080805480602002602001604051908101604052809291908181526020018280548015610b5c57602002820191906000526020600020905b815481526020019060010190808311610b48575b505050505090509550955050505050610dc5565b610bf788600081518110610b8057fe5b602001015160f81c60f81b600083600202815460018160011615610100020316600290048110610bac57fe5b815460011615610bcb5790600052602060002090602091828204019190065b9054901a7f01000000000000000000000000000000000000000000000000000000000000000218611079565b9250610c8388600181518110610c0957fe5b602001015160f81c60f81b600060018460020201815460018160011615610100020316600290048110610c3857fe5b815460011615610c575790600052602060002090602091828204019190065b9054901a7f01000000000000000000000000000000000000000000000000000000000000000218611079565b915081830193506002841015610d07576001841015610cd9576005819080600181540180825580915050906001820390600052602060002001600090919290919091505550600160075401600781905550610d06565b60068190806001815401808255809150509060018203906000526020600020016000909192909190915055505b5b8080600101915050610a89565b506005600681805480602002602001604051908101604052809291908181526020018280548015610d6457602002820191906000526020600020905b815481526020019060010190808311610d50575b5050505050915080805480602002602001604051908101604052809291908181526020018280548015610db657602002820191906000526020600020905b815481526020019060010190808311610da2575b50505050509050945094505050505b9250929050565b6060806000809050600080905060008090506000600581610ded919061113f565b506000600681610dfd919061113f565b5060008090505b6002600080546001816001161561010002031660029004905081610e2457fe5b04811015610fc457610eb387600081518110610e3c57fe5b602001015160f81c60f81b600083600202815460018160011615610100020316600290048110610e6857fe5b815460011615610e875790600052602060002090602091828204019190065b9054901a7f01000000000000000000000000000000000000000000000000000000000000000218611079565b9250610f3f87600181518110610ec557fe5b602001015160f81c60f81b600060018460020201815460018160011615610100020316600290048110610ef457fe5b815460011615610f135790600052602060002090602091828204019190065b9054901a7f01000000000000000000000000000000000000000000000000000000000000000218611079565b915081830193506002841015610fb7576001841015610f89576005819080600181540180825580915050906001820390600052602060002001600090919290919091505550610fb6565b60068190806001815401808255809150509060018203906000526020600020016000909192909190915055505b5b8080600101915050610e04565b50600560068180548060200260200160405190810160405280929190818152602001828054801561101457602002820191906000526020600020905b815481526020019060010190808311611000575b505050505091508080548060200260200160405190810160405280929190818152602001828054801561106657602002820191906000526020600020905b815481526020019060010190808311611052575b5050505050905094509450505050915091565b600080600160f81b9050600080905060008090505b600881101561113457827effffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff191683866000600181106110c857fe5b1a60f81b167effffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff191614156110fe5781806001019250505b6001837effffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff1916901b9250808060010191505061108e565b508092505050919050565b81548183558181111561116657818360005260206000209182019101611165919061116b565b5b505050565b61118d91905b80821115611189576000816000905550600101611171565b5090565b9056fea265627a7a723158202ec546e7ac67e5cbbe61213de15a1f2e2c0f0b2356ccd3801eca1ef13b0c7fe164736f6c63430005110032"
	
	
	web3.eth.defaultAccount = web3.eth.accounts[9]

	# Instantiate and deploy contract
	Greeter = web3.eth.contract(abi=abi, bytecode=bytecode)

	# Submit the transaction that deploys the contract
	tx_hash = Greeter.constructor().transact()

	# Wait for the transaction to be mined, and get the transaction receipt
	tx_receipt = web3.eth.waitForTransactionReceipt(tx_hash)

	# Create the contract instance with the newly-deployed address
	# global contract
	contract = web3.eth.contract(
		address=tx_receipt.contractAddress,
		abi=abi,
	)
	print(tx_receipt.contractAddress)
	return web3,tx_receipt.contractAddress

def storage(web3,ad):
	
	time_start = time.time()  
	
	c = web3.eth.contract(address=ad,abi=abi)
	tx_hash = c.functions.push(file1).transact()

	# # Wait for transaction to be mined...
	web3.eth.waitForTransactionReceipt(tx_hash)

	time_upload = time.time()  
	time_sum = time_upload - time_start  
	print('data upload time : ', time_sum )
	

se = '0x4717'


def search(web3, ad, query=''):
	if query != '':
		file21 =query
	else:
		file21=se
	
	contract = web3.eth.contract(address=ad,abi=abi)
	time_start = time.time()
	
	tx_hash = contract.functions.search(file21).transact()
	
	list1,list2 = contract.functions.getindex().call()
	
	# print(list1)
	# print(list2)
	
	for i in list1:
		print(file3[4*i+2:4*i+6])
	#print("0000000000000000000")
	for i in list2:
		print(file3[4*i+2:4*i+6])

	time_end = time.time()  
	time_sum = time_end - time_start  
	print('seatch time : ',time_sum)
	


def searchTopk(web3, ad,topk, query=''):
	if query != '':
		file21 =query
	else:
		file21=se
	time_start = time.time()

	contract = web3.eth.contract(address=ad,abi=abi)

	#print('numnumnum',contract.functions.get().call())
	
	tx_hash = contract.functions.topk(file21,topk).transact()
	
	list1,list2 = contract.functions.getindex().call()
	

	time_end = time.time()  
	time_sum = time_end - time_start  
	print('top'+str(topk)+'-seatch time : ',time_sum)
	return time_sum,list1,list2
	



