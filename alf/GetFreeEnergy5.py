#! /usr/bin/env python

def GetFreeEnergy5(alf_info,ms,msprof, cutb=2.0, cutc=8.0, cutx=2.0, cuts=1.0, cutc2=1.0, cutx2=0.5, cuts2=0.5):
  """
  Perform the matrix inversion to solve for optimal bias changes

  The alf.RunWham routine computes profiles and linear changes to those
  profiles in response to changes in bias parameters. It also computes a
  quadratic penalty function that penalizes deviations of the profiles
  from their average values. The second derivatives of the penalty
  function with respect to changes in the bias potential are saved as a
  matrix in analysis[i]/multisite/C.dat and the first derivatives are
  saved in analysis[i]/multisite/V.dat. This routine should be run in
  analysis[i]. 
  
  This routine adds an additional regularization term to the
  C.dat matrix and inverts the matrix to solve the linear equation
  dictating the optimal solution. Because this represents a linear
  approximation, and because unsampled states may become dominant as
  biases change, there are caps in the changes to any particular bias
  parameter. If these caps are exceed, all changes are scaled down to
  bring the largest change below these caps. The scaling is printed every
  cycle of ALF, and will be 1 or less. Smaller values indicate poorly
  converged biases. 
  
  Changes to the b, c, x, and s parameters are saved to
  b.dat, c.dat, x.dat, and s.dat in the analysis[i] directory.

  Parameters
  ----------
    Parameters
    ----------
    alf_info : dict
        Dictionary of variables ALF needs to run, including:
        - `temp` : Temperature of system.
        - `nsubs` : List specifying the number of blocks for each site.
        - `nblocks` : Total number of sub-s in the system.

    ms : int
        Flag for whether to include intersite biases:
        - `0` for no intersite biases.
        - `1` for including `c`, `x`, and `s` biases.
        - `2` for including only `c` biases.
        Typically taken from the first element of the `ntersite` list.

    msprof : int
        Flag for whether to include intersite profiles:
        - `0` for no intersite profiles.
        - `1` for including intersite profiles.
        Typically taken from the second element of the `ntersite` list.

    cutb : float, optional
        Cap for fixed bias 'b' changes (default: `2.0`).
        V = b(I)*[lambda(I)]

    cutc : float, optional
        Cap for intra-site Quadratic `c` parameter changes (default: `8.0`).
        V = c(I,J)*[lambda(I) * lambda(J)]

    cutx : float, optional
        Cap for intra-site diagonal-quadratic `x` parameter changes (default: `2.0`).
        V  = x(I,J)*[lambda(I) * lambda(J)]/[lambda(I) + 0.017]

    cuts : float, optional
        Cap for intra-site end-point `s` parameter changes (default: `1.0`).
        V = s(I,J)*[lambda(J) * (1 - exp(REF*lambda(I)))]

    cutc2 : float, optional
        Cap for inter-site `c` parameter changes (default: `1.0`).

    cutx2 : float, optional
        Cap for inter-site `x` parameter changes (default: `0.5`).

    cuts2 : float, optional
        Cap for inter-site `s` parameter changes (default: `0.5`).

  """

  import sys, os
  import numpy as np

  kT=0.001987*alf_info['temp']
  krest=1

  Emid=np.arange(1.0/800,1,1.0/400)
  Emid2=np.arange(1.0/40,1,1.0/20)

  nsubs=alf_info['nsubs']
  nblocks=alf_info['nblocks']

  b_prev=np.loadtxt('b_prev.dat')
  c_prev=np.loadtxt('c_prev.dat')
  x_prev=np.loadtxt('x_prev.dat')
  s_prev=np.loadtxt('s_prev.dat')

  b=np.zeros((1,nblocks))
  c=np.zeros((nblocks,nblocks))
  x=np.zeros((nblocks,nblocks))
  s=np.zeros((nblocks,nblocks))

  nparm=0
  for isite in range(0,len(nsubs)):
    n1=nsubs[isite]
    n2=nsubs[isite]*(nsubs[isite]-1)//2;
    for jsite in range(isite,len(nsubs)):
      n3=nsubs[isite]*nsubs[jsite]
      if isite==jsite:
        nparm+=n1+5*n2
      elif ms==1:
        nparm+=5*n3
      elif ms==2:
        nparm+=n3

  cutlist=np.zeros((nparm,))
  reglist=np.zeros((nparm,))
  n0=0
  iblock=0
  for isite in range(0,len(nsubs)):
    jblock=iblock
    n1=nsubs[isite]
    n2=nsubs[isite]*(nsubs[isite]-1)//2;
    for jsite in range(isite,len(nsubs)):
      n3=nsubs[isite]*nsubs[jsite]
      if isite==jsite:
        cutlist[n0:n0+n1]=cutb
        n0+=n1
        cutlist[n0:n0+n2]=cutc
        n0+=n2
        cutlist[n0:n0+2*n2]=cutx
        n0+=2*n2
        cutlist[n0:n0+2*n2]=cuts
        n0+=2*n2
      elif ms==1:
        cutlist[n0:n0+n3]=cutc2
        n0+=n3

        ind=n0
        for i in range(0,nsubs[isite]):
          for j in range(0,nsubs[jsite]):
            reglist[ind]=-x_prev[iblock+i,jblock+j]
            ind+=1
            reglist[ind]=-x_prev[jblock+j,iblock+i]
            ind+=1
        cutlist[n0:n0+2*n3]=cutx2
        n0+=2*n3

        ind=n0
        for i in range(0,nsubs[isite]):
          for j in range(0,nsubs[jsite]):
            reglist[ind]=-s_prev[iblock+i,jblock+j]
            ind+=1
            reglist[ind]=-s_prev[jblock+j,iblock+i]
            ind+=1
        cutlist[n0:n0+2*n3]=cuts2
        n0+=2*n3
      elif ms==2:
        cutlist[n0:n0+n3]=cutc2
        n0+=n3
      jblock+=nsubs[jsite]
    iblock+=nsubs[isite]

  if not os.path.exists('multisite/C.dat'):
    print('Error, %s/multisite/C.dat does not exist, RunWham.py probably failed, check %s/output and %s/error for clues' % (os.getcwd(),os.getcwd(),os.getcwd()))
  C=np.loadtxt('multisite/C.dat')
  for i in range(0,n0):
    C[i,i]+=krest*cutlist[i]**-2
  for i in range(0,np.shape(C)[0]):
    if C[i,i]==0:
      C[i,i]=1;
  V=np.loadtxt('multisite/V.dat')
  for i in range(0,n0):
    # Add a harmonic restraint to the total value of the x and s cross terms
    V[i]+=(krest*cutlist[i]**-2)*reglist[i]

  # coeff=C^-1*V;
  coeff=np.linalg.solve(C,V)

  coeff_orig=coeff

  scaling=1.5/np.max(np.abs(coeff[0:n0]/cutlist))
  if scaling>1:
    scaling=1
  coeff*=scaling

  print("scaling is:")
  print(scaling)

  ind=0
  iblock=0
  for isite in range(0,len(nsubs)):
    jblock=iblock
    for jsite in range(isite,len(nsubs)):
      if isite==jsite:
        for i in range(0,nsubs[isite]):
          b[0,iblock+i]=coeff[ind]
          ind+=1
        for i in range(0,nsubs[isite]):
          for j in range(i+1,nsubs[isite]):
            c[iblock+i,jblock+j]=coeff[ind]
            ind+=1
        for i in range(0,nsubs[isite]):
          for j in range(0,nsubs[isite]):
            if i != j:
              x[iblock+i,jblock+j]=coeff[ind]
              ind+=1
        for i in range(0,nsubs[isite]):
          for j in range(0,nsubs[isite]):
            if i != j:
              s[iblock+i,jblock+j]=coeff[ind]
              ind+=1
      elif ms==1:
        for i in range(0,nsubs[isite]):
          for j in range(0,nsubs[jsite]):
            c[iblock+i,jblock+j]=coeff[ind]
            ind+=1
        for i in range(0,nsubs[isite]):
          for j in range(0,nsubs[jsite]):
            x[iblock+i,jblock+j]=coeff[ind]
            ind+=1
            x[jblock+j,iblock+i]=coeff[ind]
            ind+=1
        for i in range(0,nsubs[isite]):
          for j in range(0,nsubs[jsite]):
            s[iblock+i,jblock+j]=coeff[ind]
            ind+=1
            s[jblock+j,iblock+i]=coeff[ind]
            ind+=1
      elif ms==2:
        for i in range(0,nsubs[isite]):
          for j in range(0,nsubs[jsite]):
            c[iblock+i,jblock+j]=coeff[ind]
            ind+=1
      jblock+=nsubs[jsite]
    iblock+=nsubs[isite]

  iblock=0
  for isite in range(0,len(nsubs)):
    jblock=iblock
    for jsite in range(isite,len(nsubs)):
      if isite!=jsite:
        for i in range(0,nsubs[isite]):
          b[0,iblock+i]+=c[iblock+i,jblock]
          c[iblock+i,jblock:jblock+nsubs[jsite]]-=c[iblock+i,jblock]
        for j in range(0,nsubs[jsite]):
          b[0,jblock+j]+=c[iblock,jblock+j]
          c[iblock:iblock+nsubs[isite],jblock+j]-=c[iblock,jblock+j]
      jblock+=nsubs[jsite]
    iblock+=nsubs[isite]

  iblock=0
  for isite in range(0,len(nsubs)):
    b[0,iblock:iblock+nsubs[isite]]-=b[0,iblock]
    iblock+=nsubs[isite]


  np.savetxt('b.dat',b,fmt=' %7.2f')
  np.savetxt('c.dat',c,fmt=' %7.2f')
  np.savetxt('x.dat',x,fmt=' %7.2f')
  np.savetxt('s.dat',s,fmt=' %7.2f')
