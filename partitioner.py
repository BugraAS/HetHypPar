import os 
from joblib import Parallel, delayed
import datetime

os.chdir("parts") 

def part_process(i, key, val): 
   
   f = open(f"../output/{val}","a") 
   f.write(str(datetime.datetime.now()))
   f.write("\n") ; 
   f.write(f"../matrices/{i}.mtx") 
   
   
   exit_status = os.system(f"../build/src/patpart ../matrices/{i}.mtx 16 ../{key} 1 1 {i}{val} ../results/{i}{val}") 


   if exit_status == 0:
        f.write("   Command succeeded")
   else:
        f.write("   Command failed")
   f.write("\n----------------------------------------------------\n") ; 
   f.close() 
   
   # os.system(f"mv partvec.txt {i}_16k_s1_b1_i1")
 

arr = ['cage14', 'Transport', 'nv2', 'dgreen',   'ss', 'vas_stokes_1M', 'pre2', 'twotone', 'hcircuit', 'scircuit', 'xenon2', 'lung2', 
       'stomach', 'torso1', 'torso2', 'torso3','cage12', 'cage13', 'matrix_9', 'matrix-new_3', 'barrier2-10', 'barrier2-11', 'barrier2-12', 
       'barrier2-1', 'barrier2-2', 'barrier2-3','barrier2-4', 'barrier2-9', 'ohne2', 'para-10', 'para-4', 'para-5', 'para-6', 'para-7', 
       'para-8', 'para-9', 'language', 'Hamrle3', 'dc1', 'dc2', 'dc3', 'trans4', 'trans5', 'rajat21', 'rajat23', 'rajat24', 'rajat29', 
       'rajat30',  'ASIC_320k', 'ASIC_320ks','ASIC_680k', 'ASIC_680ks',  'FEM_3D_thermal2', 'Baumann', 'crashbasis', 'majorbasis','Raj1', 
       'hvdc2', 'largebasis', 'tmt_unsym', 'thermomech_dK', 'atmosmodd', 'atmosmodj','atmosmodl','transient', 'PR02R', 'RM07R', 
       'mac_econ_fwd500', 'webbase-1M','CoupCons3D', 'ML_Laplace','power197k', 'Goodwin_095', 'Goodwin_127','imagesensor', 'power9', 
       'radiation', 'test1','ss1', 'nxp1', 'marine1']    


dict = {'benchmark0.txt':'_16k_s1_b0_i1','benchmark1.txt':'_16k_s1_b1_i1','benchmark2.txt':'_16k_s1_b2_i1'}

Parallel(n_jobs=16)( delayed(part_process)(i,key,val) for key, val in dict.items() for i in arr  ) 
     

# Segmatation fault 
# rajat31 Fullchip  Freescale2 circuit5M_dc Freescale1 memchip  patents_main