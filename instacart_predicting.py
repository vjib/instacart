#Latest updated : Dec 11, 2017

"""
We need to extract all npy files to current folder including

boughtlist.npy
trainlist.npy

npy files can be founded in data.rar. Extract them to the same folder where instacart_predicting.py is located.

"""


import numpy as np
import time
import math

"""
We may adjust this parameter due to hardware constraints in this prediction.

_TESTINDEX : index of user_id we would like to predict
_SAMPLESTART : start of index as training set
_STARTEND : end of index as training set
_THRESHOLD : The number of data that relationship among products occur in this database. The higher this value, the higher probability that we will detect strong relationship among product
_PVALUE : Statistic value to check if cumulative buying is significantly higher than average cumulative buying
"""

_TESTINDEX=25000 #1-206209 3318
_SAMPLESTART=0 #0-206208
_SAMPLEEND=5000 #2-206209, _SAMPLEEND must be higher than _SAMPLESTART
_THRESHOLD=50 #(Greater than 20 is recommened)
_PVALUE=3.09 #3.09, in this case, implies Confidence Interval at 99.99% (Greater than or equal 1.65 is recommended)


#Loading data getting from computation in instacart_preprocessing.py
t1 = int(round(time.time() * 1000))
print('Loading...')
boughtlist=np.load('boughtlist.npy') 
trainlist=np.load('trainlist.npy') 
print('Load completed!')
t2 = int(round(time.time() * 1000))
print("Time usage for loading data (second) : "+str((t2-t1)/1000))

print(boughtlist[_TESTINDEX])
print(trainlist[_TESTINDEX])

myorders=boughtlist[_TESTINDEX]

#Initialize forest tree data structure
prob_tree=[{} for _ in range(50000)]

""" 
product_id = Products that user has bought earlier. We use this to predict which products user will buy
target_product_id = Products that user buy latest (Training set)
related_product_id = Products that user has also bought earlier with product_id. We use this to find the relationship between two products user bought earlier and the next product user will buy.



product_id (25) target_product_id (4)
     \             /
      \           /
       \         /
        \       /
         \     /
          \   /
           \ /
            v
    related_product_id

This implies that 25 times of buying product_id and 4 times of related_product_id lead user to buy related_product_id next time

"""

t1 = int(round(time.time() * 1000))
for user_id in range(_SAMPLESTART,_SAMPLEEND): #Iterate in training set
	if user_id!=_SAMPLESTART and (user_id-_SAMPLESTART)%1000==0:
		print('Calculating at Round '+str((user_id-_SAMPLESTART)/1000)+'k')
	pair={}
	for idx,product_id in enumerate(boughtlist[user_id]): #Access more details in product_id for each user_id
		
		if product_id in myorders: #Check if product_id in user_id is in your order, we check this in order to reduce unnecessary computed time.

			for target_product_id in trainlist[user_id]: #Access more deatils in product_id for each user_id which is their training set
				
				for idx3,related_product_id in enumerate(boughtlist[user_id]):		

					forward=boughtlist[user_id][product_id]	#Get number of cumulative purchase for product_id
					backward=boughtlist[user_id][related_product_id] #Get number of cumulative purchase for related_product_id

					#Branching the forest tree
					if target_product_id not in prob_tree[product_id]:
						prob_tree[product_id][target_product_id]={}

					if related_product_id not in prob_tree[product_id][target_product_id]:
						#We create node in this forest tree. The first index indicates number of relationship between target_product_id bought and its pair (product_id and related_product_id). The second and third index indicate mean and variance of number of cumulative buying in product_id respectively.
						prob_tree[product_id][target_product_id][related_product_id]=[0,0,0]

					if target_product_id not in prob_tree[related_product_id]:
						prob_tree[related_product_id][target_product_id]={}

					if product_id not in prob_tree[related_product_id][target_product_id]:
						prob_tree[related_product_id][target_product_id][product_id]=[0,0,0]
		
					#We find out that if target_product_id was bought, product_id and related_product_id must be bought together recently.
					prob_tree[product_id][target_product_id][related_product_id][0]+=1
					num_cases=prob_tree[product_id][target_product_id][related_product_id][0]

					#Update mean
					prob_tree[product_id][target_product_id][related_product_id][1]=(prob_tree[product_id][target_product_id][related_product_id][1]*(num_cases-1)+forward)/num_cases
					
					#Update variance
					if num_cases>1:
						prob_tree[product_id][target_product_id][related_product_id][2]=(prob_tree[product_id][target_product_id][related_product_id][2]*(num_cases-2)/(num_cases-1))+(forward-prob_tree[product_id][target_product_id][related_product_id][1])**2/num_cases

					#We prevent duplication in the relationship of the same product_id and related_product_id 
					if product_id!=related_product_id:
						prob_tree[related_product_id][target_product_id][product_id][0]+=1
						num_cases=prob_tree[related_product_id][target_product_id][product_id][0]
						#Update mean
						prob_tree[related_product_id][target_product_id][product_id][1]=(prob_tree[related_product_id][target_product_id][product_id][1]*(num_cases-1)+backward)/num_cases
						#Update variance
						if num_cases>1:
							prob_tree[related_product_id][target_product_id][product_id][2]=(prob_tree[related_product_id][target_product_id][product_id][2]*(num_cases-2)/(num_cases-1))+(forward-prob_tree[related_product_id][target_product_id][product_id][1])**2/num_cases

t2 = int(round(time.time() * 1000))
print("Time usage for buildind forest tree (second) : "+str((t2-t1)/1000))

t1 = int(round(time.time() * 1000))

#Initialize prediction of buying array
prob_products=[0]*50000

#Consider your current order
for idx,bought_product_id in enumerate(myorders):

	#Then consider each products if it will be bought
	for predicted_product_id in range(1,50000):

		#If there is no data of predicted_product_id in the forest tree, we will ignore it to improve time cost
		if predicted_product_id in prob_tree[bought_product_id]:

			#Finally, we will look back to your items in current order. Then we form the relationship bought_product_id --> predicted_product_id --> side_product_id
			for idx2,side_product_id in enumerate(myorders):
				if side_product_id in prob_tree[bought_product_id][predicted_product_id]:

					#Get total number of relation between bought_product_id, predicted_product_id and side_product_id
					cases=prob_tree[bought_product_id][predicted_product_id][side_product_id][0]

					#Get average cumulative purchase in bought_product_id
					avg=prob_tree[bought_product_id][predicted_product_id][side_product_id][1]

					#Get S.D. of cumulative purchase in bought_product_id
					std=math.sqrt(prob_tree[bought_product_id][predicted_product_id][side_product_id][2])
					buying=myorders[bought_product_id]

					ttest=0
					#Calculating t-test
					if cases>0 and std>0:
						ttest=(cases-avg)/(std/math.sqrt(cases))

					#State condition that we will buy predicted_product_id
					if cases>=_THRESHOLD and cases>=buying and ttest>=_PVALUE:
						#print(str(bought_product_id)+' '+str(predicted_product_id)+' '+str(side_product_id)+' '+str(cases)+' '+str(buying)+' '+str(avg)+' '+str(std))
						prob_products[predicted_product_id]=1
		


t2 = int(round(time.time() * 1000))
print("Time usage for predicting purchases (second) : "+str((t2-t1)/1000))

#print(boughtlist[_TESTINDEX])
#print(trainlist[_TESTINDEX])

predicted=[]
matched=[]

#Find product which is predicted to be bought
for k in range(0,50000):
	if prob_products[k]!=0:
		predicted.append(k)

#Find correct answer for prediction
for k in range(0,len(trainlist[_TESTINDEX])):
	for j in range(0,len(predicted)):
		if trainlist[_TESTINDEX][k]==predicted[j]:
			matched.append(predicted[j])

print('Predicted purchase')
print(predicted)
print('Actual purchase')
print(sorted(trainlist[_TESTINDEX]))
print('Matched purchase')
print(sorted(matched))
print('Accuracy')
print(format(100*len(matched)/len(predicted),'.2f')+'%')
