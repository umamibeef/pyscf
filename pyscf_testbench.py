import pyscf.dft
import pyscf.gto
import numpy

benzene = [[ 'C'  , ( 4.673795 ,   6.280948 , 0.00  ) ],
           [ 'C'  , ( 5.901190 ,   5.572311 , 0.00  ) ],
           [ 'C'  , ( 5.901190 ,   4.155037 , 0.00  ) ],
           [ 'C'  , ( 4.673795 ,   3.446400 , 0.00  ) ],
           [ 'C'  , ( 3.446400 ,   4.155037 , 0.00  ) ],
           [ 'C'  , ( 3.446400 ,   5.572311 , 0.00  ) ],
           [ 'H'  , ( 4.673795 ,   7.376888 , 0.00  ) ],
           [ 'H'  , ( 6.850301 ,   6.120281 , 0.00  ) ],
           [ 'H'  , ( 6.850301 ,   3.607068 , 0.00  ) ],
           [ 'H'  , ( 4.673795 ,   2.350461 , 0.00  ) ],
           [ 'H'  , ( 2.497289 ,   3.607068 , 0.00  ) ],
           [ 'H'  , ( 2.497289 ,   6.120281 , 0.00  ) ]]


mol_hf = pyscf.gto.M(atom = benzene, basis = 'ccpvdz', symmetry = True)
mf_hf = pyscf.dft.RKS(mol_hf)
mf_hf.xc = 'b3lyp'
mf_hf = mf_hf.newton() # second-order algortihm
mf_hf.kernel()

occ_orbs = mf_hf.mo_coeff[:, mf_hf.mo_occ > 0.]
grids = pyscf.dft.gen_grid.Grids(mol_hf)
grids.build(with_non0tab=True)
weights = grids.weights
ao1 = pyscf.dft.numint.eval_ao(mol_hf, grids.coords, deriv=1, non0tab=grids.non0tab)
ts = 0.5 * np.einsum('xgp,pi,xgq,qi->g', ao1[1:], occ_orbs, ao1[1:], occ_orbs)
Ts = np.einsum('g,g->', weights, ts)