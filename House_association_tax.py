#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np                                        
import pandas as pd                                       
import matplotlib.pyplot as plt                           
data=pd.read_csv('House_prediction.csv')
                                                          
data=data.replace("-", '0')                               
data['floor']=data['floor'].astype(str).astype(int)                                           
b=data.groupby(['city']).sum()                            
                                                          
                                     
c=b.to_numpy()    

plt.rcParams['figure.figsize']=(20,10)
plt.plot(b.columns,c[0,:]/1258,color='k',marker='o') 
plt.plot(b.columns,c[1,:]/853,color='r',marker='o')  
plt.plot(b.columns,c[2,:]/1193,color='c',marker='o') 
plt.plot(b.columns,c[3,:]/1501,color='m',marker='o') 
plt.plot(b.columns,c[4,:]/5887,color='b',marker='o') 
plt.tick_params(axis ='x', rotation = 45,labelsize=20)  
plt.tick_params(axis ='y', rotation = 0,labelsize=20)
plt.legend(b.index,prop={'size':20})  
plt.title("City Wise Feature Plotting",fontsize=20)
plt.xlabel('Features',fontsize=20)
plt.ylabel('Mean',fontsize=20)
plt.show()                                           


# In[3]:


d=data[data.city=='Belo Horizonte']   
plt.subplot(3,2,1)
plt.plot(d['area'],d['fire insurance (R$)'],color='k',marker='o')
plt.title("Area vs Fire Insurance")
plt.subplot(3,2,2)
plt.plot(d['rooms'],d['fire insurance (R$)'],color='r',marker='o')
plt.title("Rooms vs Fire Insurance")
plt.subplot(3,2,3)
plt.plot(d['bathroom'],d['fire insurance (R$)'],color='c',marker='o')
plt.title("Bathroom vs Fire Insurance")
plt.subplot(3,2,4)
plt.plot(d['parking spaces'],d['fire insurance (R$)'],color='m',marker='o')
plt.title("Parking Spaces vs Fire Insurance")
plt.subplot(3,2,5)
plt.plot(d['floor'],d['fire insurance (R$)'],color='b',marker='o')
plt.title("Floor vs Fire Insurance")


plt.tight_layout()


# In[4]:


d=data[data.city=='Belo Horizonte']   
plt.subplot(3,2,1)
plt.plot(d['area'],d['hoa (R$)'],color='k',marker='o')
plt.title("Area vs HOA")
plt.subplot(3,2,2)
plt.plot(d['rooms'],d['hoa (R$)'],color='r',marker='o')
plt.title("Rooms vs HOA")
plt.subplot(3,2,3)
plt.plot(d['bathroom'],d['hoa (R$)'],color='c',marker='o')
plt.title("Bathroom vs HOA")
plt.subplot(3,2,4)
plt.plot(d['parking spaces'],d['hoa (R$)'],color='m',marker='o')
plt.title("Parking Spaces vs HOA")
plt.subplot(3,2,5)
plt.plot(d['floor'],d['hoa (R$)'],color='b',marker='o')
plt.title("Floor vs HOA")


plt.tight_layout()


# In[5]:


d=data[data.city=='Belo Horizonte']   
plt.subplot(3,2,1)
plt.plot(d['area'],d['property tax (R$)'],color='k',marker='o')
plt.title("Area vs Property Tax")
plt.subplot(3,2,2)
plt.plot(d['rooms'],d['property tax (R$)'],color='r',marker='o')
plt.title("Rooms vs Property Tax")
plt.subplot(3,2,3)
plt.plot(d['bathroom'],d['property tax (R$)'],color='c',marker='o')
plt.title("Bathroom vs Property Tax")
plt.subplot(3,2,4)
plt.plot(d['parking spaces'],d['property tax (R$)'],color='m',marker='o')
plt.title("Parking Spaces vs Property Tax")
plt.subplot(3,2,5)
plt.plot(d['floor'],d['property tax (R$)'],color='b',marker='o')
plt.title("Floor vs Property Tax")


plt.tight_layout()


# In[6]:


data=data.replace('acept','1')                            
data=data.replace('furnished','1')                        
data=data.replace('not acept','0')                        
data=data.replace('not furnished','0')                    
                                                          
                                                         
data['animal']=data['animal'].astype(str).astype(int)     
data['furniture']=data['furniture'].astype(str).astype(int)
                                                          
                                                          
                                                                                               
                                                      
for n in range(1,5):                                                         
    X=data.iloc[0:1000,n]                                     
    Y=data.iloc[0:1000,11]                                    
    m=len(X)                                                  
    mean_x=np.mean(X)                                         
    mean_y=np.mean(Y)                                         
    num=0                                                     
    den=0                                                     
    for i in range(m):                                        
        num+=(X[i]-mean_x)*(Y[i]-mean_y)                      
        den+=(X[i]-mean_x)*(X[i]-mean_x)                      
    slope=num/den                                             
    intr=mean_y-(slope*mean_x)                                
    Xp=np.array(data.iloc[1000:10692,n])                      
    Yp=intr+slope*Xp                                          
     
    plt.subplot(3,2,1)
    plt.scatter(X,Y)
    plt.plot(Xp,Yp,color='r') 
    plt.title(b.columns[n-1]+" "+"vs Rent Amount")
    plt.show()                                               
                                                          
    ss_n=0                                                    
    ss_d=0                                                    
    for i in range(m):                                        
        ss_n+=(Yp[i]-mean_y)*(Yp[i]-mean_y)                   
        ss_d+=(Y[i]-mean_y)*(Y[i]-mean_y)                     
    r2=1-(ss_n/ss_d)  
    print("Y = "+str(slope)+"* X +"+" "+str(intr) )
    print("R square" +" "+b.columns[n-1]+" "+"vs Rent Amount"+" "+":" + str(r2))  




# In[7]:


# Answer-1 : Thus according to me Porto Algere is the most possible choice


# In[8]:


# Answer-2 : Fire Insurance , Property Tax, Hoa all increases with the increase in number of bathroom.


# In[9]:


# Answer-3 : Below each graph is given the equation of regression line of that graph that I calculate using r-squared method.

