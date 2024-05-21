import os 

os.chdir("mxv-results")  

arr = ['pre2', 'twotone', 'hcircuit', 'scircuit', 'xenon2', 'lung2', 'stomach', 'torso1', 'torso2', 'torso3','cage12', 'cage13', 
       'matrix_9', 'matrix-new_3', 'barrier2-10', 'barrier2-11', 'barrier2-12', 'barrier2-1', 'barrier2-2', 'barrier2-3','barrier2-4', 
       'barrier2-9', 'ohne2', 'para-10', 'para-4', 'para-5', 'para-6', 'para-7', 'para-8', 'para-9', 'language', 'Hamrle3', 'dc1', 
       'dc2', 'dc3', 'trans4', 'trans5', 'rajat21', 'rajat23', 'rajat24', 'rajat29', 'rajat30',  'ASIC_320k', 'ASIC_320ks','ASIC_680k', 
       'ASIC_680ks',  'FEM_3D_thermal2', 'Baumann', 'crashbasis', 'majorbasis','Raj1', 'hvdc2', 'largebasis', 'tmt_unsym','thermomech_dK', 
       'atmosmodd', 'atmosmodj','atmosmodl','transient', 'PR02R', 'RM07R', 'mac_econ_fwd500', 'webbase-1M','CoupCons3D', 'ML_Laplace',
       'power197k', 'Goodwin_095', 'Goodwin_127','imagesensor', 'power9', 'radiation', 'test1','ss1', 'nxp1', 'marine1','cage14','Transport',
       'nv2', 'dgreen', 'ss', 'vas_stokes_1M']

      # Segmentation fault 
      # 'rajat31','memchip','FullChip', 'Freescale2', 'Freescale1', 'circuit5M_dc', 'patents_main',


arr_var =   ['_16k_s1_b0_i1']  #['_16k_s1_b0_i1','_16k_s1_b1_i1','_16k_s1_b2_i1']  

for i in arr: 
    for j in arr_var:
       exit_status = os.system(f"mpirun --map-by core -np 16 ../build/src/inner ../matrices/{i}.mtx  ../parts/{i}{j}  1 >> output_spring2.txt") 


spmvx_time = 0 
with open('output_spring.txt','r') as f : 
   for line in f:
        if '../parts/' in line:
            time_str = line.split(':')[2].strip()    
            if time_str:  
                spmvx_time += float(time_str)  
           
print("Total SpMVX time:", spmvx_time)