#! /usr/bin/env python

def PlotFreeEnergy5(directory=None,ntersite=[0,0]):
  """
  Plots free energy profiles for a cycle of alf

  This should be run after alf.postprocess has completed because it
  displays the free energy profiles computed by alf.RunWham.

  This version dynamically detects grid sizes from data files instead
  of using hardcoded values, and includes improved error handling.

  Parameters
  ----------
  directory : str, optional
      A string for the analysis directory of the cycle of interest. If
      blank, analysis will be performed in this directory. (default is
      None)
  ntersite : list of int, optional
      The ntersite list used during postprocessing on this cycle of alf.
      If the second element of the list is incorrect, multisite systems
      will not display correctly. (default is [0,0])
  """
  import sys, os
  import numpy as np
  import matplotlib.pyplot as plt

  # import alf
  # from prep.alf_info import alf_info
  # alf.initialize('blade')

  DIR=os.getcwd()
  if directory:
    os.chdir(directory)

  # nsubs=alf_info['nsubs']
  # nblocks=alf_info['nblocks']
  nsubs=np.loadtxt('../prep/nsubs',dtype='int',ndmin=1)
  nblocks=np.sum(nsubs)

  msprof=ntersite[1]

  iG=1
  
  # Determine grid sizes from data files
  try:
    first_1d_file = 'multisite/G'+str(iG)+'.dat'
    print(f"Trying to read: {os.path.abspath(first_1d_file)}")
    
    if not os.path.exists(first_1d_file):
      print(f"Error: File {first_1d_file} does not exist")
      print(f"Current directory: {os.getcwd()}")
      print(f"Available files in multisite/: {os.listdir('multisite/') if os.path.exists('multisite/') else 'multisite/ does not exist'}")
      grid_1d = 1024  # fallback
    else:
      first_1d_data = np.loadtxt(first_1d_file)
      if len(first_1d_data) == 0:
        print(f"Warning: File {first_1d_file} is empty, using fallback grid_1d=1024")
        grid_1d = 1024
      else:
        grid_1d = len(first_1d_data)
        print(f"Successfully read {first_1d_file}, grid_1d = {grid_1d}")
  except Exception as e:
    print(f"Error reading first 1D data file: {e}")
    grid_1d = 1024  # fallback
  
  # Find first 2D data file to determine 2D grid size
  grid_2d = None
  temp_iG = iG
  iblock = 0
  for isite in range(len(nsubs)):
    if nsubs[isite] > 2:  # 2D profiles exist for sites with >2 substituents
      # Skip 1D profiles
      temp_iG += nsubs[isite]
      # Skip transition profiles
      temp_iG += nsubs[isite] * (nsubs[isite] - 1) // 2
      # Read first 2D profile
      try:
        first_2d_data = np.loadtxt('multisite/G'+str(temp_iG)+'.dat')
        grid_2d = int(np.sqrt(len(first_2d_data)))
        break
      except:
        pass
    else:
      # Skip profiles for this site
      temp_iG += nsubs[isite] + nsubs[isite] * (nsubs[isite] - 1) // 2
    iblock += nsubs[isite]
  
  if grid_2d is None:
    grid_2d = 32  # fallback
    
  print(f"Detected grid sizes: 1D={grid_1d}, 2D={grid_2d}")
  
  # Create dynamic grids based on detected sizes
  Emid=np.arange(1.0/(2*grid_1d), 1, 1.0/grid_1d)
  Emid2=np.arange(1.0/(2*grid_2d), 1, 1.0/grid_2d)
  EmidX,EmidY=np.meshgrid(Emid2,Emid2)
  iF=1

  G1={}
  G12={}
  G2={}
  G11={}
  iblock=0
  for isite in range(len(nsubs)):
    jblock=iblock
    for jsite in range(isite,len(nsubs)):
      if isite==jsite:

        plt.figure(iF)
        iF=iF+1
        LEG=[]
        for i in range(nsubs[isite]):
          LEG.append(str(i+1))
          G1[i+iblock]=np.loadtxt('multisite/G'+str(iG)+'.dat')
          
          # Handle inf values
          inf_count = np.sum(np.isinf(G1[i+iblock]))
          if inf_count > 0:
            print(f"Site {isite+1}, substituent {i+1}: Found {inf_count} inf values, replacing with NaN")
            G1[i+iblock][np.isinf(G1[i+iblock])] = np.nan
          
          # Debug output for each profile
          print(f"Site {isite+1}, substituent {i+1}: Reading G{iG}.dat")
          print(f"  Data shape: {G1[i+iblock].shape}")
          finite_data = G1[i+iblock][np.isfinite(G1[i+iblock])]
          if len(finite_data) > 0:
            print(f"  Data range (finite): {np.min(finite_data):.3f} to {np.max(finite_data):.3f}")

          else:
            print(f"  No finite data found!")
          
          iG=iG+1
          # plot(Emid,G1{i+iblock})
          plt.plot(Emid,G1[i+iblock])
          plt.xlabel('lambda')
          plt.ylabel('Free energy [kcal/mol]')
        plt.title('Site %d:   1D profiles' % (isite+1,))
        plt.legend(LEG)
        
        # Focus on lambda range 0.1 to 0.9 (central transition region)
        # Find the corresponding indices for lambda 0.1 to 0.9
        lambda_start_idx = int(0.1 * grid_1d)
        lambda_end_idx = int(0.9 * grid_1d)
        
        print(f"Focusing on lambda range 0.1-0.9 (indices {lambda_start_idx}:{lambda_end_idx})")
        
        # Collect finite data only from the 0.1-0.9 lambda range
        transition_data = []
        for i in range(nsubs[isite]):
          data_subset = G1[i+iblock][lambda_start_idx:lambda_end_idx]
          finite_subset = data_subset[np.isfinite(data_subset)]
          if len(finite_subset) > 0:
            transition_data.extend(finite_subset)
        
        if len(transition_data) > 0:
          transition_data = np.array(transition_data)
          
          # Set y-limits based on the 0.1-0.9 lambda range data
          y_min = float(np.min(transition_data))
          y_max = float(np.max(transition_data))
          
          # Add some padding (5% on each side)
          padding = (y_max - y_min) * 0.05
          y_min -= padding
          y_max += padding
          
          print(f"Site {isite+1} lambda 0.1-0.9 range: {y_min:.3f} to {y_max:.3f}")
          
          plt.ylim(y_min, y_max)
        else:
          print(f"Site {isite+1}: No finite data in lambda 0.1-0.9 range!")
          plt.ylim(-0.5, 2.5)

        pltfg=plt.figure(iF)
        iF=iF+1
        for i in range(nsubs[isite]):
          G12[i+iblock]={}
          for j in range((i+1),nsubs[isite]):
            G12[i+iblock][j+iblock]=np.loadtxt('multisite/G'+str(iG)+'.dat')
            iG=iG+1
            # plot(Emid,G12{i+iblock,j+iblock})
            plt.subplot(nsubs[isite]-1,nsubs[isite]-1,i*(nsubs[isite]-1)+j)
            plt.plot(Emid,G12[i+iblock][j+iblock])
        plt.suptitle('Site %d:   Transition profiles' % (isite+1,))
        pltfg.supxlabel('Column+1 on at x=0')
        pltfg.supylabel('Row substituent on at x=1')

        if nsubs[isite]>2:
          plt.figure(iF)
          iF=iF+1
          for i in range(nsubs[isite]):
            G2[i+iblock]={}
            for j in range((i+1),nsubs[isite]):
              G2[i+iblock][j+iblock]=np.reshape(np.loadtxt('multisite/G'+str(iG)+'.dat'),(grid_2d,grid_2d))
              iG=iG+1
              # surf(Emid2,Emid2,G2{i+iblock,j+iblock})
              pltax=plt.subplot(nsubs[isite]-1,nsubs[isite]-1,i*(nsubs[isite]-1)+j,projection='3d')
              pltax.plot_surface(EmidX,EmidY,G2[i+iblock][j+iblock])
          plt.suptitle('Site %d:   2D profiles' % (isite+1,))

      elif msprof:

        plt.figure(iF)
        iF=iF+1
        for i in range(nsubs[isite]):
          G11[i+iblock]={}
          for j in range(nsubs[jsite]):
            G11[i+iblock][j+jblock]=np.reshape(np.loadtxt('multisite/G'+str(iG)+'.dat'),(grid_2d,grid_2d))
            iG=iG+1
            # surf(Emid2,Emid2,G11{i+iblock,j+jblock})
            pltax=plt.subplot(nsubs[isite],nsubs[jsite],i*nsubs[jsite]+j+1,projection='3d')
            pltax.plot_surface(EmidX,EmidY,G11[i+iblock][j+jblock])
        plt.suptitle('Sites %d and %d:   2D coupling profiles' % (isite+1,jsite+1))

      jblock=jblock+nsubs[jsite]
    iblock=iblock+nsubs[isite]

  plt.show()
  plt.savefig('FreeEnergy.png')
  os.chdir(DIR)

if __name__ == '__main__':
  import sys
  if len(sys.argv)==1:
    PlotFreeEnergy5()
  elif len(sys.argv)==2:
    PlotFreeEnergy5(sys.argv[1])
  elif len(sys.argv)==3:
    PlotFreeEnergy5(ntersite=[int(sys.argv[1]),int(sys.argv[2])])
  elif len(sys.argv)==4:
    PlotFreeEnergy5(sys.argv[1],ntersite=[int(sys.argv[2]),int(sys.argv[3])])
