#! /usr/bin/env python

def GetEnergy(alf_info, Fi, Ff, skipE=1):
    """
    Compute energies for wham/mbar reweighting with shift file support

    This routine computes relative bias energies from simulations for
    reweighting with wham/mbar. It supports loading shift files from
    analysis directories to adjust biases based on replica index.

    Shift files are loaded from analysis[i]/nbshift/ or ../nbshift/ (fallback):
    - b_shift.dat, c_shift.dat, x_shift.dat, s_shift.dat (required)
    - b_fix_shift.dat, c_fix_shift.dat, x_fix_shift.dat, s_fix_shift.dat (optional)

    Biases are computed as: bias_prev + shift * (k - ncentral) + fix_shift

    Outputs:
    - Lambda/Lambda[i].dat - alchemical trajectories
    - Energy/ESim[i].dat - bias energies for each simulation
    - G_imp_shifts/shifts_sim[i].dat - G_imp shift information

    Parameters
    ----------
    alf_info : dict
        Dictionary of variables alf needs to run
    Fi : int
        The first cycle of alf to include in analysis (inclusive)
    Ff : int
        The final cycle of alf to include in analysis (inclusive)
    skipE : int, optional
        Only analyze frames with index modulus skipE equal to skipE-1.
        (default is 1 to analyze all frames)
    """
    import sys, os
    import numpy as np

    NF = Ff - Fi + 1

    nblocks = alf_info['nblocks']
    nsubs = alf_info['nsubs']
    nreps = alf_info['nreps']
    ncentral = alf_info['ncentral']
    

    def load_shift_file(analysis_dir, filename, default_value=0.0):
        """Load shift file, first checking analysis_dir/nbshift, then ../nbshift"""
        # Try analysis_dir/nbshift first
        local_path = os.path.join(analysis_dir, 'nbshift', filename)
        if os.path.exists(local_path):
            print(f"Loading {filename} from {local_path}")
            return np.loadtxt(local_path)
        
        # Fall back to ../nbshift
        fallback_path = os.path.join('../nbshift', filename)
        if os.path.exists(fallback_path):
            print(f"Loading {filename} from {fallback_path} (fallback)")
            return np.loadtxt(fallback_path)
        
        # Default value if neither exists
        print(f"{filename} not found, using {default_value}")
        return default_value

    Lambda = []
    b = []
    c = []
    x = []
    s = []

    for i in range(NF):
        analysis_dir = f'../analysis{Fi+i}'
        data_dir = os.path.join(analysis_dir, 'data')
        
        if not os.path.isdir(data_dir):
            print(f"Warning: Directory {data_dir} not found")
            continue

        # Load shift files for this analysis directory
        try:
            b_shift = load_shift_file(analysis_dir, 'b_shift.dat')
            c_shift = load_shift_file(analysis_dir, 'c_shift.dat')
            x_shift = load_shift_file(analysis_dir, 'x_shift.dat')
            s_shift = load_shift_file(analysis_dir, 's_shift.dat')
        except Exception as e:
            print(f"Error loading required shift files for {analysis_dir}: {e}")
            continue

        # Load fix_shift files, defaulting to 0 if not found
        b_fix_shift = load_shift_file(analysis_dir, 'b_fix_shift.dat', 0.0)
        c_fix_shift = load_shift_file(analysis_dir, 'c_fix_shift.dat', 0.0)
        x_fix_shift = load_shift_file(analysis_dir, 'x_fix_shift.dat', 0.0)
        s_fix_shift = load_shift_file(analysis_dir, 's_fix_shift.dat', 0.0)

        # print(f"Checking directory: {data_dir}")

        lambda_files = [f for f in os.listdir(data_dir) if f.startswith('Lambda.') and f.endswith('.dat')]
        lambda_files.sort()

        for lambda_file in lambda_files:
            file_path = os.path.join(data_dir, lambda_file)
            print(f"Loading file: {file_path}")
            try:
                Lambda.append(np.loadtxt(file_path)[(skipE-1)::skipE,:])
                
                # Extract j and k from the filename (assuming format Lambda.j.k.dat)
                j, k = map(int, lambda_file.split('.')[1:3])
                b_old = np.loadtxt(os.path.join(analysis_dir, 'b_prev.dat'))
                b.append(b_old + b_shift * (k - ncentral) + b_fix_shift)
                c_old = np.loadtxt(os.path.join(analysis_dir, 'c_prev.dat'))
                c.append(c_old + c_shift * (k - ncentral) + c_fix_shift)
                x_old = np.loadtxt(os.path.join(analysis_dir, 'x_prev.dat'))
                x.append(x_old + x_shift * (k - ncentral) + x_fix_shift)
                s_old = np.loadtxt(os.path.join(analysis_dir, 's_prev.dat'))
                s.append(s_old + s_shift * (k - ncentral) + s_fix_shift)

                for idx in range(len(b_old)):
                    try:
                        b_shift_val = float(b_shift[idx]) if hasattr(b_shift, '__getitem__') else float(b_shift)
                    except (TypeError, IndexError):
                        b_shift_val = float(b_shift)
                    
                    try:
                        b_fix_shift_val = float(b_fix_shift[idx]) if hasattr(b_fix_shift, '__getitem__') else float(b_fix_shift)
                    except (TypeError, IndexError):
                        b_fix_shift_val = float(b_fix_shift)
                    
                    print(f"Shifted b[{idx}] from {b_old[idx]:.2f} to {b[-1][idx]:.2f} using k={k}, ncentral={ncentral}, b_shift={b_shift_val:.2f}, b_fix_shift={b_fix_shift_val:.2f}")
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")

    if not os.path.isdir('Lambda'):
        os.mkdir('Lambda')
    if not os.path.isdir('Energy'):
        os.mkdir('Energy')

    total_simulations = len(Lambda)
    # print(f"Total simulations: {total_simulations}")

    if total_simulations == 0:
        print("Error: No Lambda files found.")
        return

    E = [[] for _ in range(total_simulations)]

    try:
        for i in range(total_simulations):
            for j in range(total_simulations):
                bi, ci, xi, si = b[i], c[i], x[i], s[i]
                Lj = Lambda[j]
                E[i].append(np.reshape(np.dot(Lj,-bi),(-1,1)))
                E[i][-1] += np.sum(np.dot(Lj,-ci)*Lj,axis=1,keepdims=True)
                E[i][-1] += np.sum(np.dot(1-np.exp(-5.56*Lj),-xi)*Lj,axis=1,keepdims=True)
                E[i][-1] += np.sum(np.dot(Lj/(Lj+0.017),-si)*Lj,axis=1,keepdims=True)
    except Exception as e:
        print(f"Error during energy calculation: {e}")
        return

    for i in range(total_simulations):
        Ei = E[total_simulations-1][i]
        for j in range(total_simulations):
            Ei = np.concatenate((Ei,E[j][i]),axis=1)
        np.savetxt(f'Energy/ESim{i+1}.dat', Ei, fmt='%12.5f')

    for i in range(total_simulations):
        Li = Lambda[i]
        np.savetxt(f'Lambda/Lambda{i+1}.dat', Li, fmt='%10.6f')

    # Create G_imp shift information file
    # Since energy uses negative terms (np.dot(Lj,-bi)), G_imp shifts need opposite signs
    print("\nCreating G_imp shift information...")
    
    if not os.path.isdir('G_imp_shifts'):
        os.mkdir('G_imp_shifts')
    
    # We need to track j,k values for each simulation as we process Lambda files
    simulation_jk_map = {}
    sim_idx = 0
    
    for i in range(NF):
        analysis_dir = f'../analysis{Fi+i}'
        data_dir = os.path.join(analysis_dir, 'data')
        
        if not os.path.isdir(data_dir):
            continue
            
        lambda_files = [f for f in os.listdir(data_dir) if f.startswith('Lambda.') and f.endswith('.dat')]
        lambda_files.sort()
        
        for lambda_file in lambda_files:
            try:
                # Extract j and k from filename
                j, k = map(int, lambda_file.split('.')[1:3])
                simulation_jk_map[sim_idx] = (j, k, i)  # Store j, k, and analysis_index
                sim_idx += 1
            except (ValueError, IndexError):
                print(f"Warning: Could not parse j,k from {lambda_file}")
                simulation_jk_map[sim_idx] = (1, sim_idx, i)  # Fallback
                sim_idx += 1
    
    # For each simulation, save the G_imp shifts (opposite of energy shifts)
    for sim_idx in range(total_simulations):
        if sim_idx not in simulation_jk_map:
            continue
            
        j, k, analysis_idx = simulation_jk_map[sim_idx]
        analysis_dir = f'../analysis{Fi + analysis_idx}'
        
        try:
            # Load the shift files used for this analysis directory  
            b_shift = load_shift_file(analysis_dir, 'b_shift.dat', 0.0)
            b_fix_shift = load_shift_file(analysis_dir, 'b_fix_shift.dat', 0.0)
            
            # For G_imp shifts, we need the opposite of energy shifts
            # Energy: -b_shift * (k - ncentral) - b_fix_shift
            # G_imp: +b_shift * (k - ncentral) + b_fix_shift
            
            # Calculate G_imp shifts (opposite of energy shifts)
            if hasattr(b_shift, '__getitem__') and hasattr(b_shift, '__len__'):
                gimp_shift = np.array(b_shift)  # Array of shifts per block
                gimp_fix_shift = np.array(b_fix_shift)  # Array of fix shifts per block
                num_blocks = len(gimp_shift)
            else:
                gimp_shift = np.array([float(b_shift)] * nblocks)  # Single value to array
                gimp_fix_shift = np.array([float(b_fix_shift)] * nblocks)
                num_blocks = nblocks
            
            # Save shift information for this simulation
            # Save as simple text file for WHAM to read
            with open(f'G_imp_shifts/shifts_sim{sim_idx + 1}.dat', 'w') as f:
                f.write(f"# G_imp shift information for simulation {sim_idx + 1}\n")
                f.write(f"# j: {j}, k: {k}, ncentral: {ncentral}\n")
                f.write(f"# Formula: gimp_fix_shift[block] + gimp_shift[block] * (k - ncentral)\n")
                for block_idx in range(num_blocks):
                    total_shift = float(gimp_fix_shift[block_idx]) + float(gimp_shift[block_idx]) * (k - ncentral)
                    f.write(f"{total_shift:.6f}\n")
            
            print(f"Created G_imp shift file for simulation {sim_idx + 1} (j={j}, k={k})")
                
        except Exception as e:
            print(f"Warning: Could not create G_imp shifts for simulation {sim_idx + 1}: {e}")

    print("Energy calculation completed successfully.")