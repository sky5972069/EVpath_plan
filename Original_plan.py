import numpy as np
import time
import sys
import matplotlib.pylab as plt
import matplotlib.pyplot as mp
import xlrd
import xlwt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d
import matplotlib.cm as cm
from pylab import mpl
import time, timeit
from decimal import Decimal
import pandas as pd
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体

		#################################################################
		#distance function
		#################################################################



class Node:
	def __init__(self, parent, n_id, fn):
		self.parent = parent
		self.id = n_id
		self.fn = fn


def cal_price(path_array, pathlist, new_timelist, waitlist, price_array):
	startpoint = 80
	endpoint = 80
	pathwaitlist = []
	pathpricelist = []		
	for pi in range(len(pathlist)-1):


		pathpricelist.append(price_array[pathlist[pi]-1])
		pathwaitlist.append(waitlist[pathlist[pi]-1])

	
	pricenew = 0
	# 
	# print('value1', valuenew)
	for  pi in range(len(pathlist)-1):
				
		endpoint = startpoint+new_timelist[pi]
		if (int(startpoint)==int(endpoint)):
				pricenew += pathpricelist[pi][int(startpoint)]*(endpoint-startpoint)
		else:
			for i in range(int(startpoint),int(endpoint)+1):
			# print(i)
				if(i==int(startpoint)):
					pricenew += pathpricelist[pi][i]*(int(startpoint)+1-startpoint)
							
					# print('1',valuenew, new_timelist,b*charge)
				elif(i==int(endpoint)):
					pricenew += pathpricelist[pi][i]*(endpoint-int(endpoint))
							# print('2',valuenew, new_timelist)
				else:
					pricenew += pathpricelist[pi][i]
					# print('3',valuenew, new_timelist)
		if ((pi+1)<len(pathwaitlist)):
			startpoint = endpoint+path_array[pathlist[pi]-1][pathlist[pi+1]-1]+pathwaitlist[pi+1]

	return pricenew





def aim(path_array, price_array, pathlist, waitlist, soc0=16, socn=4, a=1, b=1, loss=1, charge=4, markovlen=100, k=0.96):
	T_d = 0                              	 	  # time travelling
	T_c = 0                              		  # time charging
	T_w = 0                             		  # time waiting
	cost = 0									  # charging cost
	pathpricelist = []                  		  # price of the point chosen
	pathwaitlist = []



	current_timelist = []                         # charging time
	best_timelist = []
	new_timelist = []
	mincost_list = []
	currentcost_list = []


	currentcost = 9999 
	mincost = 9999
	

	new_soc = soc0
	for pi in range(len(pathlist)-1):

		T_d += path_array[pathlist[pi]-1][pathlist[pi+1]-1]
		pathpricelist.append(price_array[pathlist[pi]-1])
		pathwaitlist.append(waitlist[pathlist[pi]-1])
		# current_timelist.append(round((path_array[pathlist[pi]-1][pathlist[pi+1]-1]*loss)/charge))
		# best_timelist.append(round(path_array[pathlist[pi]-1][pathlist[pi+1]-1]*loss/charge))
		# new_timelist.append(round(path_array[pathlist[pi]-1][pathlist[pi+1]-1]*loss/charge))
		randseed = (max(0,((path_array[pathlist[pi]-1][pathlist[pi+1]-1]*loss-new_soc)/charge))+(24-new_soc)/charge)/2
		current_timelist.append(randseed)
		best_timelist.append(randseed)
		new_timelist.append(randseed)
		new_soc += new_timelist[pi]*charge-path_array[pathlist[pi]-1][pathlist[pi+1]-1]*loss		
	
	# print(pathwaitlist)
	# print(pathpricelist)
	# T_c = (socn-soc0+loss*T_d)/charge

	# print(current_timelist)

	alpha = k
	t2 = (1,30)
	markovlen = markovlen
	t = t2[1]
	step = 0.1
	new_soc = soc0
	# print(new_timelist)
	# print(pathpricelist)
	flag = 0
	while t > t2[0]:
		for i in np.arange(markovlen):
			#################################################################
			#Produce new solutions
			#################################################################
			while (True):
				new_soc = soc0
				# print('123')
				for station_no in range(len(pathlist)-1):
					
					
						new_timelist[station_no] = np.random.uniform(max(0,current_timelist[station_no]-step*(t**0.7),((path_array[pathlist[station_no]-1][pathlist[station_no+1]-1]*loss-new_soc)/charge)),
							min(current_timelist[station_no]+step*(t**0.5),(24-new_soc)/charge))
						new_soc += new_timelist[station_no]*charge-path_array[pathlist[station_no]-1][pathlist[station_no+1]-1]*loss
					# else:
					# 	new_timelist[station_no] = current_timelist[station_no] .rand()-0.5)*(t**0.5/10)

				# print(new_timelist)
				# print(current_timelist)
				

				#################################################################
				#Judge if the solusion is valid
				#
				# print(new_timelist)
				count = 0
				current_soc = soc0
				# print(new_timelist)
				for pi in range(len(new_timelist)):
					
					current_soc += new_timelist[pi]*charge
					
					if (current_soc > path_array[pathlist[pi]-1][pathlist[pi+1]-1]*loss) and (current_soc<=24):
						count += 1
					current_soc -= path_array[pathlist[pi]-1][pathlist[pi+1]-1]*loss
					# print(current_soc)
				if (current_soc>=socn):
					count += 1



				# print('c1',count)
				for i in new_timelist:
					if(i >= 0):
						count += 1
				# print('c2',count)

				# print(count)

				if(count == 2*len(new_timelist)+1):
					break

				# print(count)
				# print('456')

			# print(newtimelist)

			####################################################################
			#Calculate new Value
			####################################################################

			startpoint = 80
			endpoint = 80
			
			T_c = sum(new_timelist)

			# not useful waiting time
			#

			T_w = sum(pathwaitlist)
			valuenew = a*(T_d+T_c+T_w)
			pricenew = 0
			# 
			# print('value1', valuenew)
			for  pi in range(len(pathlist)-1):
				
				endpoint = startpoint+new_timelist[pi]
				if (int(startpoint)==int(endpoint)):
					valuenew += b*charge*pathpricelist[pi][int(startpoint)]*(endpoint-startpoint)
				else:
					for i in range(int(startpoint),int(endpoint)+1):
						# print(i)
						if(i==int(startpoint)):
							pricenew += pathpricelist[pi][i]*(int(startpoint)+1-startpoint)
							
							# print('1',valuenew, new_timelist,b*charge)
						elif(i==int(endpoint)):
							pricenew += pathpricelist[pi][i]*(endpoint-int(endpoint))
							# print('2',valuenew, new_timelist)
						else:
							pricenew += pathpricelist[pi][i]
							# print('3',valuenew, new_timelist)
				if ((pi+1)<len(pathwaitlist)):
					startpoint = endpoint+path_array[pathlist[pi]-1][pathlist[pi+1]-1]+pathwaitlist[pi+1]
				# print('bb',b*charge*pathpricelist[pi][i],valuenew)
			# print('value2',valuenew)
				# print(valuenew)
			valuenew += b*pricenew
			######################################################################
			#judgement for acceptance
			
			if valuenew < currentcost: #接受该解
				#更新solutioncurrent 和solutionbest			######################################################################

				currentcost = valuenew
				current_timelist = new_timelist.copy()
				currentcost_list.append(currentcost)

				if valuenew < mincost:
					mincost = valuenew
					bestprice = pricenew
					best_timelist = new_timelist.copy()
					best_a = a*(T_d+T_c+T_w)
					best_b = mincost-a*(T_d+T_c+T_w)
			else:#按一定的概率接受该解
				rands = np.random.rand()
				# print(rands)
				# print(np.exp(-(valuenew-currentcost)/(t/100)))
				if rands < np.exp(-(valuenew-currentcost)/(t/2000)):
					flag+=1
					# print(flag)

					currentcost = valuenew
					current_timelist = new_timelist.copy()
					currentcost_list.append(currentcost)
				else:
					currentcost_list.append(currentcost_list[-1])
				
		# print('a',a*(T_d+T_c+T_w))
		# print('b',mincost-a*(T_d+T_c+T_w))
		# print('bt',mincost,best_timelist)
		# print(charge)

		t = alpha*t


		# print (t,besttimelist) #程序运行时间较长，打印t来监视程序进展速度




### the process of Simulated annealing algorithm 

	# L = 0                                         #loop
	# while (L < 0):
	# 	for pi in range(len(pathlist)-1):
	# 		current_soc = current_soc-loss*path_array[pathlist[i]-1][pathlist[i+1]-1]

	# if pathlist[-1]==30:
	# 	output_time = xlwt.open_workbook("C:/电动汽车项目/项目论文/转投论文/时间序列.xlsx")	
	# 	output_time.write(best_timelist)
	# 	output_time.close()
	# min_xaxis = [i for i in range(len(mincost_list))]
	# current_xaxis = [i for i in range(len(currentcost_list))]
	# if(pathlist[-1]==30):
	# 	# plt.plot(min_xaxis, mincost_list, label='最优值')
	# 	plt.plot(current_xaxis, currentcost_list, label='新值')
	# 	# plt.legend()
		
	# 	plt.ylabel('最优目标函数值')
	# 	plt.xlabel('更新次数')
	# 	plt.show()
	return(mincost,best_timelist,best_a,best_b,bestprice)

			


######################################################################
#A* Algorithm
######################################################################


def planning(path_array, price_array, waitlist, distance=0, soc0=16, socn=4, a=1, b=1, loss=1, charge=4, r=20):
	
	
	num = int(path_array.size**0.5)       # num of points
	target_station = len(path_array)
	stations = []
	for i in range(num):
		stations.append(Node(0, i+1, path_array[0][i]))
	stations[0].fn = 999
	minfn = 9999
	final_node = stations[-1]
	openlist = [stations[0]]    				  # save best path

	closelist = []                        # save accessed points

	passedarray = []

	while(True):
				
		######################################################################
		# find the node with the smallest fn
		#
		for i in range(len(openlist)):
			
			# print(i)
			templist_node = [openlist[i]]
			templist = [openlist[i].id]

			while templist_node[0] != stations[0]:
				templist_node.insert(0, templist_node[0].parent)
				templist.insert(0,templist_node[0].id)

			if templist in passedarray:
				# print('passed',templist)
				continue

			elif (openlist[i].fn < minfn):
				minfn = openlist[i].fn
				current_node = i                     #当前节点


		# print("n", openlist[current_node].id)

		templist_node = [openlist[current_node]]
		templist = [openlist[current_node].id]

		while templist_node[0] != stations[0]:
			templist_node.insert(0, templist_node[0].parent)
			templist.insert(0,templist_node[0].id)
		print(templist)
		passedarray.append(templist)
		# print(passedarray)

		minfn = 9999

		######################################################################
		# output
		#
		if (openlist[current_node] == final_node):
			print("fuck")
			best_path = [openlist[current_node]]
			best_path_id = []
			while (best_path[0] != stations[0]):
				best_path.insert(0, best_path[0].parent)
			for i in best_path:
				best_path_id.append(i.id)
			best_value,best_timelist,best_period,best_price_b,best_price = aim(path_array=path_array, price_array=price_array,pathlist=best_path_id, waitlist=waitlist, soc0=soc0, socn=socn, a=a, b=b, loss=loss, charge=charge)

			return (best_path_id,best_timelist,best_value,best_period,best_price_b,best_price)

		######################################################################
		# calculate the fn for its sons and update
		#
		for i in range(num):
			

			# print(i)


			if (path_array[templist[-1]-1][i]>40) or (templist[-1]==i+1):
				
				continue

			if (len(templist)>=2) and (templist[-2]==i+1):
				continue
			# if (stations[i].id != openlist[current_node].id):
				#for one son, get its current path in templist
				
				

			# while templist_node[0] != stations[0]:
			# 	templist_node.insert(0, templist_node[0].parent)
			# 	templist.insert(0,templist_node[0].id)


			else:
				templist_node.append(stations[i])
				templist.append(stations[i].id)
				print("templist", templist)
				templist_value,best_timelist,aaa,bbb,ccc = aim(path_array=path_array, price_array=price_array,pathlist=templist, waitlist=waitlist, soc0=soc0, socn=socn, a=a, b=b, loss=loss, charge=charge)
				#calculate
				fx = distance[i]/2/(r+15) + templist_value
				# print('fx', fx)
				############################################################################
				#update openlist and closelist
				#
				if stations[i] in openlist:
					# print("stat1")
					if (fx<stations[i].fn):
						# print("update")
						stations[i].parent = openlist[current_node]
						stations[i].fn = fx

				elif stations[i] in closelist:
					# print("stat2")
					continue

				elif (stations[i] not in openlist) and (stations[i] not in closelist):
					# print("stat3")
					stations[i].parent = openlist[current_node]
					stations[i].fn = fx
					openlist.append(stations[i])
				
				templist_node.pop()
				templist.pop()

		#outpout the station fn

		# for i in range(len(openlist)):
		# 	print(openlist[i].fn)


		######################################################################
		# calculate the fn for its sons and update
		#

		closelist.append(openlist[current_node])
		# print(openlist)
		# print(closelist)

		# openlist = [1,2,3,4]
		# return aim(path_array=path_array, price_array=price_array,pathlist=openlist, soc0=soc0, socn=socn, a=1, b=1, loss=2, charge=5)



######################################################################
# get the maxnum of a list
#
def max_(target):

	maxnum = 0
	for i in range(len(target)):
		thismax = max(target[i])
		if (thismax > maxnum):
			maxnum = thismax

	return (maxnum)



if (__name__=='__main__'):
	
	
	mypath = np.array([[0,7,6,12,99,99],[7,0,5,8,99,11],[6,5,0,10,12,8],[12,8,10,0,8,10],[99,99,12,8,0,12],[99,11,8,10,12,0]])
	myprice = np.array([[1 for i in range(500)] for j in range(500)])
	myprice[1] = [20 for i in range(500)]
	myprice[2] = [20 for i in range(500)]
	myprice[3] = [4 for i in range(500)]
	mycar_p = 16
	mycharge_p = [10,10,10,10,10]
	mydistance = [14,11,8,10,1,0]
	ak = max(mydistance)
	bk = max_(myprice)



	
	Super_Path_Array = np.array([[0,10,7,12,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[10,0,99,5,7,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[7,99,0,99,99,7,14,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[12,5,99,0,7,99,8,99,99,12,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[99,7,99,7,0,99,99,99,99,99,6,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[99,99,7,99,99,0,9,99,99,99,99,99,13,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[99,99,14,8,99,9,0,4,7,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[99,99,99,99,99,99,4,0,6,99,99,5,99,99,99,12,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[99,99,99,99,99,99,7,6,0,99,99,99,99,99,8,8,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[99,99,99,12,99,99,99,99,99,0,99,99,99,99,3,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[99,99,99,99,6,99,99,99,99,99,0,99,99,5,5,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[99,99,99,99,99,99,99,5,99,99,99,0,9,99,99,99,12,3,9,99,99,99,99,99,99,99,99,99,99,99],
		[99,99,99,99,99,13,99,99,99,99,99,9,0,99,99,99,99,99,7,99,99,99,99,99,99,99,99,99,99,99],
		[99,99,99,99,99,99,99,99,99,99,5,99,99,0,5,99,99,99,99,5,3,99,99,99,99,99,99,99,99,99],
		[99,99,99,99,99,99,99,99,8,3,5,99,99,5,0,10,99,99,99,99,99,10,99,99,99,99,99,99,99,99],
		[99,99,99,99,99,99,99,12,8,99,99,99,99,99,10,0,5,99,99,99,16,99,9,99,99,16,99,99,99,99],
		[99,99,99,99,99,99,99,99,99,99,99,12,99,99,99,5,0,99,99,99,99,99,99,99,9,99,99,99,99,99],
		[99,99,99,99,99,99,99,99,99,99,99,3,99,99,99,99,99,0,99,99,99,99,99,5,16,99,99,99,99,99],
		[99,99,99,99,99,99,99,99,99,99,99,9,7,99,99,99,99,99,0,99,99,99,99,9,99,99,99,99,99,99],
		[99,99,99,99,99,99,99,99,99,99,99,99,99,5,99,99,99,99,99,0,4,99,99,99,99,99,7,99,99,99],
		[99,99,99,99,99,99,99,99,99,99,99,99,99,3,99,16,99,99,99,4,0,5,99,99,99,99,99,9,99,99],
		[99,99,99,99,99,99,99,99,99,99,99,99,99,99,10,99,99,99,99,99,5,0,8,99,99,99,99,7,99,99],
		[99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,9,99,99,99,99,99,8,0,99,99,99,99,99,7,7],
		[99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,5,9,99,99,99,99,0,14,99,99,99,99,99],
		[99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,9,16,99,99,99,99,99,14,0,7,99,99,99,99],
		[99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,16,99,99,99,99,99,99,99,99,7,0,99,99,99,6],
		[99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,7,99,99,99,99,99,99,0,6,99,99],
		[99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,9,7,99,99,99,99,6,0,7,99],
		[99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,7,99,99,99,99,7,0,7],
		[99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,7,99,99,6,99,99,7,0]])


	Final_Path_Array = np.array([[0,4,6,4,99,6,7,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[4,0,8,6,4,8,6,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[6,8,0,6,9,3,8,99,99,99,11,8,8,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[4,6,6,0,4,6,4,6,9,10,10,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[99,4,9,4,0,8,5,5,99,99,10,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[6,8,3,6,99,0,6,99,6,99,8,8,7,8,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[7,6,8,4,5,6,0,5,5,7,9,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[99,99,99,6,5,99,5,0,8,5,8,10,9,6,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[99,99,99,9,99,6,5,8,0,99,3,5,4,4,99,99,6,7,7,10,12,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[99,99,99,10,99,99,7,5,99,0,7,99,99,7,2,99,99,4,11,7,10,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99],


		[99,99,11,10,10,8,9,8,3,7,0,6,4,1,8,99,6,7,6,9,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[99,99,8,99,99,8,99,10,5,99,6,0,3,99,99,3,3,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[99,99,8,99,99,7,99,9,4,99,4,3,0,4,99,99,3,4,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[99,99,99,99,99,8,99,6,4,7,1,99,4,0,8,99,7,7,5,8,9,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[99,99,99,99,99,99,99,99,99,2,8,99,99,8,0,99,99,3,10,6,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[99,99,99,99,99,99,99,99,99,99,99,3,99,99,99,0,3,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[99,99,99,99,99,99,99,99,6,99,6,3,3,7,99,3,0,99,4,11,11,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[99,99,99,99,99,99,99,99,7,4,7,99,4,7,3,99,99,0,7,4,6,9,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[99,99,99,99,99,99,99,99,7,11,6,99,99,5,10,99,4,7,0,99,8,11,10,99,12,11,12,13,12,14,99,99,99,99,99,99],
		[99,99,99,99,99,99,99,99,10,7,9,99,99,8,6,99,11,4,99,0,3,6,99,8,8,99,10,10,11,99,99,99,99,99,99,99],


		[99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,0,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,0,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,0,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,0,99,99,99,99,99,99,99,99,99,99,99,99],
		[99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,0,99,99,99,99,99,99,99,99,99,99,99],
		[99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,0,99,99,99,99,99,99,99,99,99,99],
		[99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,0,99,99,99,99,99,99,99,99,99],
		[99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,0,99,99,99,99,99,99,99,99],
		[99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,0,99,99,99,99,99,99,99],
		[99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,0,99,99,99,99,99,99],


		[99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,0,99,99,99,99,99],
		[99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,0,99,99,99,99],
		[99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,0,99,99,99],
		[99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,0,99,99],
		[99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,0,99],
		[99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,0],])




	Final_Path_Array = np.array([[0,4,6,4,99,6,7,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[4,0,8,6,4,8,6,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[6,8,0,6,9,3,8,99,99,99,11,8,8,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[4,6,6,0,4,6,4,6,9,10,10,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[99,4,9,4,0,8,5,5,99,99,10,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[6,8,3,6,99,0,6,99,6,99,8,8,7,8,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[7,6,8,4,5,6,0,5,5,7,9,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[99,99,99,6,5,99,5,0,8,5,8,10,9,6,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[99,99,99,9,99,6,5,8,0,99,3,5,4,4,99,99,6,7,7,10,12,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[99,99,99,10,99,99,7,5,99,0,7,99,99,7,2,99,99,4,11,7,10,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99],


		[99,99,11,10,10,8,9,8,3,7,0,6,4,1,8,99,6,7,6,9,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[99,99,8,99,99,8,99,10,5,99,6,0,3,99,99,3,3,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[99,99,8,99,99,7,99,9,4,99,4,3,0,4,99,99,3,4,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[99,99,99,99,99,8,99,6,4,7,1,99,4,0,8,99,7,7,5,8,9,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[99,99,99,99,99,99,99,99,99,2,8,99,99,8,0,99,99,3,10,6,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[99,99,99,99,99,99,99,99,99,99,99,3,99,99,99,0,3,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[99,99,99,99,99,99,99,99,6,99,6,3,3,7,99,3,0,99,4,11,11,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[99,99,99,99,99,99,99,99,7,4,7,99,4,7,3,99,99,0,7,4,6,9,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[99,99,99,99,99,99,99,99,7,11,6,99,99,5,10,99,4,7,0,99,8,11,10,99,12,11,12,13,12,14,99,99,99,99,99,99],
		[99,99,99,99,99,99,99,99,10,7,9,99,99,8,6,99,11,4,99,0,3,6,99,8,8,99,10,10,11,99,99,99,99,99,99,99],


		[99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,0,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,0,99,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,0,99,99,99,99,99,99,99,99,99,99,99,99,99],
		[99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,0,99,99,99,99,99,99,99,99,99,99,99,99],
		[99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,0,99,99,99,99,99,99,99,99,99,99,99],
		[99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,0,99,99,99,99,99,99,99,99,99,99],
		[99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,0,99,99,99,99,99,99,99,99,99],
		[99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,0,99,99,99,99,99,99,99,99],
		[99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,0,99,99,99,99,99,99,99],
		[99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,0,99,99,99,99,99,99],


		[99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,0,99,99,99,99,99],
		[99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,0,99,99,99,99],
		[99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,0,99,99,99],
		[99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,0,99,99],
		[99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,0,99],
		[99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,0],])






	######################################################################	
	# waiting time generating
	#


	Super_Wait_Array = [0 for i in range(30)]


	# Super_Wait_Array[14] = 2 
	# Super_Wait_Array[15] = 2 
	# Super_Wait_Array[8] = 2 
	# # Super_Wait_Array[22] = 2 
	# Super_Wait_Array[21] = 2 




	# Super_Price_Array = [[1.0 for i in range(360)]for j in range(30)]
	# for j in range(len(Super_Price_Array)):
	# 	for i in range(len(Super_Price_Array[0])):

	# 		if (i<105) or (i>=315) or (165<=i<195):
	# 			Super_Price_Array[j][i] = 1.5
	# 		elif(105<=i<165):
	# 			Super_Price_Array[j][i] = 2
	# for i in range(30):
	# 	# for j in range(360):
	# 	Super_Price_Array[i]=Super_Price_Array[i]*5

	
	Super_Distance = [51,44,45,39,37,40,31,28,24,28,28,33,45,23,25,16,21,29,36,27,20,25,7,27,13,6,20,14,7,0]
	# print(len(Super_Distance))
	Super_Charge = [10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]
	
	
	axis_x_price = [i for i in range(96)]

	######################################################################	
	# load and price data
	#
	data = xlrd.open_workbook("C:/电动汽车项目/项目论文/转投论文/基本负荷.xlsx")

	table1 = data.sheets()[0]
	x1 = []
	y1 = []

	for i in range(table1.nrows):
		x1.append(table1.cell_value(i,0)*4)


	for i in range(table1.nrows):
		# if table1.cell_value(i,1)<700:
			y1.append(table1.cell_value(i,1))
		# else:
		# 	y1.append(1000)



	table2 = data.sheets()[1]
	x2 = []
	y2 = []

	for i in range(table2.nrows):
		x2.append(table2.cell_value(i,0)*4)

	for i in range(table2.nrows):
		if table2.cell_value(i,1)<700:
			y2.append(table2.cell_value(i,1))
		else:
			y2.append(1000)


	######################################################################	
	# different pricse
	#

	# normal prices
	Super_Price_Array = [[i for i in y1] for j in range(len(Super_Path_Array))]
	for i in range(len(Super_Price_Array)):
		Super_Price_Array[i] = [(3*int(x/100)+3) for x in Super_Price_Array[i]]

	print(Super_Price_Array)


	# special prices

	S_Super_Price_Array = [[i for i in y1] for j in range(len(Super_Path_Array))]
	for i in range(len(Super_Price_Array)):
		if i in [1,3,4,8,9,10]:
			S_Super_Price_Array[i] = [(1.5*int(x/100)+1.5) for x in S_Super_Price_Array[i]]
			print(1)
		else:
			S_Super_Price_Array[i] = [(int(x/100)+1.5) for x in S_Super_Price_Array[i]]


	

	# reduced prices

	R_Super_Price_Array = [[i for i in y2] for j in range(len(Super_Path_Array))]
	for i in range(len(Super_Price_Array)):
		if i not in [1,3,4,6,8,9,13,14,15,21,22]:
			R_Super_Price_Array[i] = [(1.2*int(x/100)+3) for x in R_Super_Price_Array[i]]
		else:
			R_Super_Price_Array[i] = [x for x in Super_Price_Array[i]]
	# print(R_Super_Price_Array)






	over_path = 60
	all_price = 39*0.8*((sum(Super_Price_Array[0][80:])+sum(Super_Price_Array[0][:30]))/(len(Super_Price_Array[0][80:])+len(Super_Price_Array[0][:30])))/4
	print(all_price)
	# print(Super_Price_Array[0])
	
	for i in range(30):

		Super_Price_Array[i]=Super_Price_Array[i]*2
		R_Super_Price_Array[i]=R_Super_Price_Array[i]*2
	# fig, ax1 = plt.subplots() # 使用subplots()创建窗口
	
	# ax1.plot(x1,Super_Price_Array[0][:96], c='orangered',label='充电价格', linewidth = 1, marker='.') #绘制折线图像1,圆形点，标签，线宽
	# mp.legend(loc=2)#显示折线的意义
	# ax2 = ax1.twinx() # 创建第二个坐标轴
	# # plt.ylim(-1, 3)
	
	# ax2.plot(x1, y1,  c='blue',label='基本负荷', linewidth = 1, marker='.') #同上

	# ax1.set_xlabel('时间/15分钟') #与原始matplotlib设置参数略有不同，使用自己下载的中文宋体，参数位置不可改变
	# ax1.set_ylabel('充电价格/元')

	# ax2.set_ylabel('基本负荷/kW')
	# mp.gcf().autofmt_xdate()#自动适应刻度线密度，包括x轴，y轴

	# mp.legend(loc=1)#显示折线的意义
	# plt.show()

	# path, time, value, period, price = planning(path_array=Super_Path_Array, price_array=Super_Price_Array,
	# 			distance=Super_Distance,soc0=16, socn=4, a=1/over_path,b=1/all_price, charge=2.5)

	# print(planning(path_array=Super_Path_Array, price_array=Super_Price_Array,distance=Super_Distance,a=0.5/over_path,b=1.5/all_price))
	
	######################################################################	
	# outputdata generating and saving
	
	# for i in range(5):
	# 	outputfile = open("C:/电动汽车项目/项目论文/转投论文/输出文档1.txt",'a')
	# 	for j in range(2):
	# 		path, time, value, period, price, bprice = planning(path_array=Super_Path_Array, price_array=R_Super_Price_Array,
	# 			waitlist=Super_Wait_Array, distance=Super_Distance, soc0=16, socn=4, a=(2-0.1*i)/over_path,b=0.1*i/all_price, charge=4)
	# 		outputfile.write(str(i)+','+str(path)+','+str(time)+','+str(sum(time))+','+str(value)+','+str(period)+','+str(price)+','+str(bprice)+'\n')
	# 	outputfile.write('\n')
	# outputfile.close()


	######################################################################	
	# changing the proportion of distance
	#
	# valuelist = []
	# timespent = []
	# for i in range(86,100):
	# 	# 
	# 	time0 = time.time()
	# 	path, timetime, value, period, price = planning(path_array=Super_Path_Array, price_array=Super_Price_Array,
	# 		waitlist=Super_Wait_Array, distance=Super_Distance, soc0=16, socn=4, a=1/over_path,b=1/all_price, charge=4,r = i)
	# 	time1 = time.time()
	# 	valuelist.append(value)
	# 	timespent.append(time1-time0)
	# 	print(valuelist)
	# 	print(timespent)
	# 	outputfile = open("C:/电动汽车项目/项目论文/转投论文/输出文档1.txt",'a')
	# 	outputfile.write(str(value)+','+str(time1-time0))
	# 	outputfile.write('\n')
	# 	outputfile.close()



	#####################################################################	
	# calculate control group 
	
	# outputfile = open("C:/电动汽车项目/项目论文/转投论文/输出文档1.txt",'a')
	# control_price = cal_price(Super_Path_Array, [1, 2, 5, 11, 14, 21, 22, 23, 30],[1.848,1.881,1.323,1.607,2.219,0.860,0.005,0.007],Super_Wait_Array,R_Super_Price_Array)
	# print(control_price)
	# outputfile.write(str(control_price))
	# outputfile.write('\n')
	# outputfile.close()


	######################################################################	
	# pic drawing for heatmap
	#
	# mklen_range = range(2,101,2)
	# k_range = range(90,100)

	# mklen_k_list = [[0 for i in k_range] for j in mklen_range]
	# for mklen in range(len(mklen_range)):
	# 	for kk in range(len(k_range)):
	# 		for i in range(1):	
	# 			time0 = time.time()
	# 			best_value,best_timelist,best_period,best_price = aim(path_array=Super_Path_Array, price_array=Super_Price_Array,pathlist=[1, 4, 7, 9, 16, 23, 30], soc0=16, socn=8, a=1/over_path, b=1/all_price, loss=1, charge=2.5, markovlen=mklen_range[mklen], k=k_range[kk]/100)
	# 			# print(best_value,best_timelist,best_period,best_price)
	# 			# mklen_k_list[mklen][kk]+=best_value
	# 			time1 = time.time()
	# 			# print(time1-time0)
	# 			mklen_k_list[mklen][kk]+=(time1-time0)

	# print(mklen_k_list)

	# fig = plt.figure()
	# ax = fig.gca(projection='3d')
	# X = np.arange(0, len(k_range), 1)
	# Y = np.arange(0, len(mklen_range), 1)
	# X, Y = np.meshgrid(X, Y)
	# # ax.grid(False)



	# z=np.array(mklen_k_list)

	# # plt.xlim((0, 160))
	# # plt.ylim((0, 160))
	# plt.xlabel('k')
	# plt.ylabel('m')
	# ax.set_zlabel('运算时间/s')
	# #0.857  0.86352
	# plt.xticks([0, 2, 4, 6, 8],  ['0.9', '0.92', '0.94', '0.96', '0.98'])
	# plt.yticks([0, 10, 20, 30, 40, 50],  ['0', '20', '40', '60', '80', '100'])
	# # plt.colorbar()
	# # Plot the surface.
	# surf = ax.plot_surface(X, Y, z, cmap=cm.coolwarm,
	#                        linewidth=0, antialiased=False)
	# plt.show()


	data = pd.read_csv('C:/mycode/tensorflow/lstm/output_price.csv', header=None, encoding = "gbk")
	# print(names[2])
	print(data[0])
	starttime = 72
	














