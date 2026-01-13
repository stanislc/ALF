#! /usr/bin/env python

def GetFreeEnergy5(alf_info, ms=0, msprof=0, cutb=2.0, cutc=8.0, cutx=0.2, cuts=0.2,
                   cutc2=1.0, cutx2=0.5, cuts2=0.5, calc_omega=True, calc_chi=True,
                   bias_boost_strength=0.0):
  """
  Perform the matrix inversion to solve for optimal bias changes

  The alf.RunWham routine computes profiles and linear changes to those
  profiles in response to changes in bias parameters. It also computes a
  quadratic penalty function that penalizes deviations of the profiles
  from their average values. The second derivatives of the penalty
  function with respect to changes in the bias potential are saved as a
  matrix in analysis[i]/multisite/C.dat and the first derivatives are
  saved in analysis[i]/multisite/V.dat. This routine should be run in
  analysis[i]. This routine adds an additional regularization term to the
  C.dat matrix and inverts the matrix to solve the linear equation
  dictating the optimal solution. Because this represents a linear
  approximation, and because unsampled states may become dominant as
  biases change, there are caps in the changes to any particular bias
  parameter. If these caps are exceed, all changes are scaled down to
  bring the largest change below these caps. The scaling is printed every
  cycle of ALF, and will be 1 or less. Smaller values indicate poorly
  converged biases. Changes to the b, c, x, and s parameters are saved to
  b.dat, c.dat, x.dat, and s.dat in the analysis[i] directory.

  Parameters
  ----------
  alf_info : dict
      Dictionary of variables alf needs to run
  ms : int
      Flag for whether to include intersite biases. 0 for no, 1 for c, x,
      and s biases, 2 for just c biases. Typically taken from the first
      element of ntersite list.
  msprof : int
      Flag for whether to include intersite profiles. 0 for no, 1 for yes.
      Typically taken from the second element of ntersite list.
  calc_omega : bool
      Whether to calculate x (omega) parameters. Default True.
  calc_chi : bool
      Whether to calculate s (chi) parameters. Default True.
  bias_boost_strength : float
      Strength of bias boost for undersampled sites. 0 disables boost.
      Higher values (1-5) give stronger push away from zero bias.
  """

  import sys, os
  import numpy as np

  kT=0.001987*alf_info['temp']
  krest=1

  Emid=np.arange(1.0/800,1,1.0/1000)
  Emid2=np.arange(1.0/40,1,1.0/20)

  nsubs=alf_info['nsubs']
  nblocks=alf_info['nblocks']

  b_prev=np.loadtxt('b_prev.dat')
  c_prev=np.loadtxt('c_prev.dat')
  x_prev=np.loadtxt('x_prev.dat')
  s_prev=np.loadtxt('s_prev.dat')

  # Ensure b_prev is 2D
  if b_prev.ndim == 1:
    b_prev = b_prev.reshape(1, -1)

  b=np.zeros((1,nblocks))
  c=np.zeros((nblocks,nblocks))
  x=np.zeros((nblocks,nblocks))
  s=np.zeros((nblocks,nblocks))

  # Count parameters (conditional on calc_omega/calc_chi)
  nparm=0
  for isite in range(0,len(nsubs)):
    n1=nsubs[isite]
    n2=nsubs[isite]*(nsubs[isite]-1)//2
    for jsite in range(isite,len(nsubs)):
      n3=nsubs[isite]*nsubs[jsite]
      if isite==jsite:
        nparm += n1 + n2  # b + c always
        if calc_omega:
          nparm += 2*n2  # x terms
        if calc_chi:
          nparm += 2*n2  # s terms
      elif ms==1:
        nparm += n3  # c2 always
        if calc_omega:
          nparm += 2*n3  # x2 terms
        if calc_chi:
          nparm += 2*n3  # s2 terms
      elif ms==2:
        nparm += n3

  cutlist=np.zeros((nparm,))
  reglist=np.zeros((nparm,))

  # Check if bias boost is needed
  bias_magnitude = np.linalg.norm(b_prev)
  min_boost_threshold = 0.02 * kT
  needs_bias_boost = bias_boost_strength > 0 and bias_magnitude < min_boost_threshold

  if needs_bias_boost:
    print(f"Applying bias boost (strength={bias_boost_strength})")

  n0=0
  iblock=0
  for isite in range(0,len(nsubs)):
    jblock=iblock
    n1=nsubs[isite]
    n2=nsubs[isite]*(nsubs[isite]-1)//2
    for jsite in range(isite,len(nsubs)):
      n3=nsubs[isite]*nsubs[jsite]
      if isite==jsite:
        # b parameters
        cutlist[n0:n0+n1]=cutb

        # Apply bias boost if needed
        if needs_bias_boost:
          ind = n0
          for i in range(nsubs[isite]):
            if abs(b_prev[0, iblock+i]) < 1e-6:
              # Boost toward non-zero values for non-reference states
              if i > 0:
                reglist[ind] = -bias_boost_strength * kT
            else:
              reglist[ind] = -b_prev[0, iblock+i]
            ind += 1

        n0+=n1

        # c parameters
        cutlist[n0:n0+n2]=cutc
        n0+=n2

        # x parameters (conditional)
        if calc_omega:
          cutlist[n0:n0+2*n2]=cutx
          n0+=2*n2

        # s parameters (conditional)
        if calc_chi:
          cutlist[n0:n0+2*n2]=cuts
          n0+=2*n2

      elif ms==1:
        # c2 parameters
        cutlist[n0:n0+n3]=cutc2
        n0+=n3

        # x2 parameters (conditional)
        if calc_omega:
          ind=n0
          for i in range(0,nsubs[isite]):
            for j in range(0,nsubs[jsite]):
              reglist[ind]=-x_prev[iblock+i,jblock+j]
              ind+=1
              reglist[ind]=-x_prev[jblock+j,iblock+i]
              ind+=1
          cutlist[n0:n0+2*n3]=cutx2
          n0+=2*n3

        # s2 parameters (conditional)
        if calc_chi:
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
    if np.isclose(C[i,i], 0):
      C[i,i]=1
  V=np.loadtxt('multisite/V.dat')
  for i in range(0,n0):
    # Add a harmonic restraint to the total value of the x and s cross terms
    V[i]+=(krest*cutlist[i]**-2)*reglist[i]

  # coeff=C^-1*V;
  try:
    coeff = np.linalg.solve(C, V)
  except np.linalg.LinAlgError:
    print("Warning: C is singular or ill-conditioned, using least squares solution.")
    coeff, _, _, _ = np.linalg.lstsq(C, V, rcond=None)


  scaling=1.5/np.max(np.abs(coeff[0:n0]/cutlist))
  if scaling>1 or np.isnan(scaling) or np.isinf(scaling):
    scaling=1
  coeff*=scaling

  print("scaling is:")
  print(scaling)

  # Extract coefficients back to matrices
  ind=0
  iblock=0
  for isite in range(0,len(nsubs)):
    jblock=iblock
    for jsite in range(isite,len(nsubs)):
      if isite==jsite:
        # b coefficients
        for i in range(0,nsubs[isite]):
          b[0,iblock+i]=coeff[ind]
          ind+=1
        # c coefficients
        for i in range(0,nsubs[isite]):
          for j in range(i+1,nsubs[isite]):
            c[iblock+i,jblock+j]=coeff[ind]
            ind+=1
        # x coefficients (conditional)
        if calc_omega:
          for i in range(0,nsubs[isite]):
            for j in range(0,nsubs[isite]):
              if i != j:
                x[iblock+i,jblock+j]=coeff[ind]
                ind+=1
        # s coefficients (conditional)
        if calc_chi:
          for i in range(0,nsubs[isite]):
            for j in range(0,nsubs[isite]):
              if i != j:
                s[iblock+i,jblock+j]=coeff[ind]
                ind+=1
      elif ms==1:
        # c2 coefficients
        for i in range(0,nsubs[isite]):
          for j in range(0,nsubs[jsite]):
            c[iblock+i,jblock+j]=coeff[ind]
            ind+=1
        # x2 coefficients (conditional)
        if calc_omega:
          for i in range(0,nsubs[isite]):
            for j in range(0,nsubs[jsite]):
              x[iblock+i,jblock+j]=coeff[ind]
              ind+=1
              x[jblock+j,iblock+i]=coeff[ind]
              ind+=1
        # s2 coefficients (conditional)
        if calc_chi:
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


  np.savetxt('b.dat',b,fmt=' %7.4f')
  np.savetxt('c.dat',c,fmt=' %7.4f')
  np.savetxt('x.dat',x,fmt=' %7.4f')
  np.savetxt('s.dat',s,fmt=' %7.4f')
