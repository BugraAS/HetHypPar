import os 

f = open("matrices.txt", "r")

#os.chdir("matrices") 

name_list= [] 
 
for x in f:
   
    arr = x.split() 
    
    name_list.append(arr[0])
    
    if os.path.exists('/matrices/'+arr[0]+'.mtx') == False :
          
      os.system(f"wget https://suitesparse-collection-website.herokuapp.com/MM/{arr[1]}/{arr[0]}.tar.gz") 
       
      name = arr[0] + '.tar.gz'
      os.system(f"tar xvzf {name}")
      os.system(f"rm {name}") 
      os.system(f"mv {name[0:-7]}/{name[0:-7]}.mtx matrices/") 
      os.system(f"rm -r {name[0:-7]}") 
      
     
      
f.close() 
print(name_list)
