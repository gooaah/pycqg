import ase.io
from .molcrystal import atoms2molcryst
import numpy as np
from ase import Atoms
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.optimize import root
import sys

# (Under development) Compress molecular crystals to the closest-packed form.

def lda_mol(centers, rltPos, cell, ratio, coefEps=1e-3, ratioEps=1e-1, singleLDA=False):
    """
    Find the direction with largest vaccum gaps.
    """
    cartPos = []
    classes = []
    n = 0
    for cen, rlt in zip(centers, rltPos):
        for x in range(2):
            for y in range(2):
                for z in range(2):
                    n += 1
                    offset = np.dot([x,y,z], cell)
                    pos = cen + rlt + offset
                    cartPos.extend(pos.tolist())
                    classes.extend([n]*len(pos))
    
    clf = LinearDiscriminantAnalysis(n_components=3)
    clf.fit(cartPos, classes)
    print("")
    print(f"Variance ratio: {clf.explained_variance_ratio_}")

    stress = np.zeros((3,3))
    if not singleLDA:
        for i in range(3):
            ldaVec = clf.scalings_[:,i]
            ldaVec = ldaVec/np.linalg.norm(ldaVec)
            oneStress = np.outer(ldaVec, ldaVec)
            stress += oneStress*clf.explained_variance_ratio_[i]
        # stress += oneStress
    else:
        ldaVec = clf.scalings_[:,0]
        ldaVec = ldaVec/np.linalg.norm(ldaVec)
        stress = np.outer(ldaVec, ldaVec)

    # print(ldaVec)

    # print(cell)
    f_h = np.dot(np.linalg.inv(cell.T), stress)
    print(f"f_h: {f_h}")


    sclF = np.dot(f_h, np.linalg.inv(cell))
    # parameters of cubic equation
    p3 = -1 * np.linalg.det(sclF)
    p2 = sclF[0,0]*sclF[1,1] + sclF[1,1]*sclF[2,2] + sclF[0,0]*sclF[2,2]
    - sclF[0,1]*sclF[1,0] - sclF[0,2]*sclF[2,0] - sclF[1,2]*sclF[2,1]
    p1 = -1 * np.trace(sclF)
    p0 = 1 - ratio

    coefs = np.array([p3,p2,p1,p0])
    print(f"coefs: {coefs}")
    coefs[np.abs(coefs) < coefEps] = 0
    print(f"Reduced coefs: {coefs}")
    r = np.roots(coefs)
    # print(r)
    c = r[r>0].min()
    initVol = np.linalg.det(cell)
    rdcCell = cell - c*f_h
    rdcVol = np.linalg.det(cell - c*f_h)
    print("Initial Volume: {}".format(initVol))
    print("Reduced Volume: {}".format(rdcVol))
    print("Target ratio: {}, Real ratio: {}".format(ratio, rdcVol/initVol))
    if abs(ratio-rdcVol/initVol) < ratioEps:
        return rdcCell
    else:
        return None

def compress_mol_crystal(molC, minRatio, bondRatio=1.1, nsteps=5):
    partition = [set(p) for p in molC.partition]
    ratioArr = np.linspace(1, minRatio, nsteps+1)
    ratioArr = ratioArr[1:]/ratioArr[:-1]
    inMolC = molC

    for ratio in ratioArr:
        centers = inMolC.get_centers()
        rltPos = inMolC.get_rltPos()
        cell = inMolC.get_cell()
        rdcCell = lda_mol(centers, rltPos, cell, ratio)
        outMolC = inMolC.copy()
        if rdcCell is None:
            print("Too different ratio")
            return inMolC
        else:
            outMolC.set_cell(rdcCell)
            testMolC = atoms2molcryst(outMolC.to_atoms(), bondRatio)
            if False in [set(p) in partition for p in testMolC.partition]:
                print('Overlap between molecules')
                return inMolC
            else:
                inMolC = outMolC

    return outMolC




