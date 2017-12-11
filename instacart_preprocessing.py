#Latest updated : Dec 11, 2017

"""
We need to extract all csv files to current folder including

products.csv
order_products__prior.csv
order_products__train.csv
orders.csv

CSV files can be downloaded from https://www.instacart.com/datasets/grocery-shopping-2017. Put them all to the same folder where instacart_preprocessing.py is located.

"""

from openpyxl import load_workbook
import csv
import numpy as np
import time
import random

_NUM_USERS=206209
_NUMS_PRODUCTS=49688

print('loading...')

file  = open('products.csv', "r" ,encoding="utf8")
read = csv.reader(file)

products=[{} for _ in range(_NUMS_PRODUCTS)]
num_products=0
is_header=1


for product_id in range(_NUMS_PRODUCTS) :
	products[product_id]['freq']=0 # Define how much this product is bought for each users

#read orders data (prior) to list
orders=[[] for _ in range(35000000)]

t1 = int(round(time.time() * 1000))
with open('order_products__prior.csv', "r" ,encoding="utf8") as f: 
	next(f)
	for line in f:
		
		line = line.split(",")
		order_id=int(line[0])
		product_id=line[1]
		seq=int(line[2])
		
		orders[order_id].append(product_id)
	


#read orders data (train) to list
with open('order_products__train.csv', "r" ,encoding="utf8") as f: 
	next(f)
	for line in f:

		line = line.split(",")
		order_id=int(line[0])
		product_id=line[1]
		seq=int(line[2])
		
		orders[order_id].append(product_id)

t2 = int(round(time.time() * 1000))
print('Load completed!')
print("Time usage for loading prior data (second) : "+str((t2-t1)/1000))

#read users data (train) to list	
users=[{} for _ in range(_NUM_USERS)]
boughtlist=[{} for _ in range(_NUM_USERS)]
trainlist=[[] for _ in range(_NUM_USERS)]

t1 = int(round(time.time() * 1000))


total_train=0

with open('orders.csv', "r" ,encoding="utf8") as f: 
	next(f)
	temp_dow=0.0
	temp_hod=0.0
	temp_dsp=0.0
	temp_user=1
	for line in f:
		line = line.split(",")
		order_id=int(line[0])
		user_id=int(line[1])
		eval=line[2]
		order_number=float(line[3])
		dow=float(line[4])
		hod=float(line[5])
		dsp=0

		if user_id!=temp_user:
			users[temp_user]['dow']=float(temp_dow)
			users[temp_user]['hod']=float(temp_hod)
			users[temp_user]['dsp']=float(temp_dsp)
			
			temp_dow=0.0
			temp_hod=0.0
			temp_dsp=0.0
			temp_user=user_id

		temp_dow=float(temp_dow*(order_number-1)+dow)/(order_number)
		temp_hod=float(temp_hod*(order_number-1)+hod)/(order_number)
		temp_dsp=float(temp_dsp*(order_number-1)+dsp)/(order_number)

		if eval=="prior":
			for item in orders[order_id]: # Count how many this product is bought by this user
				product_id=int(item)
				if not product_id in boughtlist[user_id-1]:
					boughtlist[user_id-1][product_id]=0
				boughtlist[user_id-1][product_id]+=1


				products[product_id-1]['freq']+=1
		else:
			for item in orders[order_id]: # Collect the latest product bought by this user
				product_id=int(item)
				trainlist[user_id-1].append(product_id)
			total_train+=1

t2 = int(round(time.time() * 1000))
print("Time usage for prepcrocessing (second): "+str((t2-t1)/1000))

#print(boughtlist[1])
#print(trainlist[1])

#Save data for prediction step
np.save('boughtlist.npy', boughtlist) #File collecting prior set for every users
np.save('trainlist.npy', trainlist) #File collecting training set for every users

print('Complete preprocessing!')