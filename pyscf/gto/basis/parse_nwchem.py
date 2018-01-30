#!/usr/bin/env python
# -*- coding: utf-8
# Author: Qiming Sun <osirpt.sun@gmail.com>
#
# parse NWChem format
#

import re
import numpy

MAXL = 10
SPDF = ('S', 'P', 'D', 'F', 'G', 'H', 'I', 'K', 'L', 'M')
MAPSPDF = {'S': 0,
           'P': 1,
           'D': 2,
           'F': 3,
           'G': 4,
           'H': 5,
           'I': 6,
           'K': 7,
           'L': 8,
           'M': 9,
          }

BASIS_SET_DELIMITER = re.compile('# *BASIS SET.*\n')
ECP_DELIMITER = re.compile('\n *ECP *\n')

def parse(string, symb=None, optimize=False):
    '''Parse the basis text which is in NWChem format, return an internal
    basis format which can be assigned to :attr:`Mole.basis`
    Lines started with # are ignored.

    Kwargs:
        optimize : Optimize basis contraction.  Convert the segment contracted
            basis to the general contracted basis.
    '''
    from pyscf.gto.mole import _std_symbol
    if symb is not None:
        symb = _std_symbol(symb)
        string = _search_seg(re.split(BASIS_SET_DELIMITER, string), symb)
        if string is None:
            raise KeyError('Basis not found for %s' % symb)

    bastxt = []
    for dat in string.splitlines():
        x = dat.split('#')[0].strip().upper()  # Use # to start comments
        if (x and not x.startswith('END') and not x.startswith('BASIS')):
            bastxt.append(x)
    return _parse(bastxt, optimize)

def load(basisfile, symb, optimize=False):
    return _parse(search_seg(basisfile, symb), optimize)

def parse_ecp(string, symb=None):
    from pyscf.gto.mole import _std_symbol
    if symb is not None:
        symb = _std_symbol(symb)
        raw_data = string.splitlines()
        for i, dat in enumerate(raw_data):
            dat0 = dat.split(None, 1)
            if dat0 and dat0[0] == symb:
                break
        if i+1 == len(raw_data):
            raise KeyError('ECP not found for %s' % symb)
        seg = []
        for dat in raw_data[i:]:
            dat = dat.strip().upper()
            if dat: # remove empty lines
                if ((dat[0].isalpha() and dat.split(None, 1)[0] != symb.upper())):
                    break
                else:
                    seg.append(dat)
    else:
        seg = string.splitlines()

    ecptxt = []
    for dat in seg:
        x = dat.split('#')[0].strip().upper()
        if (x and not x.startswith('END') and not x.startswith('ECP')):
            ecptxt.append(x)
    return _parse_ecp(ecptxt)

def load_ecp(basisfile, symb):
    return _parse_ecp(search_ecp(basisfile, symb))

def search_seg(basisfile, symb):
    from pyscf.gto.mole import _std_symbol
    symb = _std_symbol(symb)
    with open(basisfile, 'r') as fin:
        fdata = re.split(BASIS_SET_DELIMITER, fin.read())
    dat = _search_seg(fdata, symb)
    if dat is None:
        return []
    else:
        return [x.upper() for x in dat.splitlines() if x and 'END' not in x]

def _search_seg(raw_data, symb):
    for dat in raw_data[1:]:
        dat0 = dat.split(None, 1)
        if dat0 and dat0[0] == symb:
            return dat

def search_ecp(basisfile, symb):
    from pyscf.gto.mole import _std_symbol
    symb = _std_symbol(symb)
    with open(basisfile, 'r') as fin:
        fdata = re.split(ECP_DELIMITER, fin.read())
    fdata = fdata[1].splitlines()
    for i, dat in enumerate(fdata):
        dat0 = dat.split(None, 1)
        if dat0 and dat0[0] == symb:
            break
    seg = []
    for dat in fdata[i:]:
        dat = dat.strip().upper()
        if dat:  # remove empty lines
            if ((dat[0].isalpha() and dat.split(None, 1)[0] != symb.upper())):
                return seg
            else:
                seg.append(dat)
    return []


def convert_basis_to_nwchem(symb, basis):
    '''Convert the internal basis format to NWChem format string'''
    from pyscf.gto.mole import _std_symbol
    res = []
    symb = _std_symbol(symb)

    # pass 1: comment line
    ls = [b[0] for b in basis]
    nprims = [len(b[1:]) for b in basis]
    nctrs = [len(b[1])-1 for b in basis]
    prim_to_ctr = {}
    for i, l in enumerate(ls):
        if l in prim_to_ctr:
            prim_to_ctr[l][0] += nprims[i]
            prim_to_ctr[l][1] += nctrs[i]
        else:
            prim_to_ctr[l] = [nprims[i], nctrs[i]]
    nprims = []
    nctrs = []
    for l in set(ls):
        nprims.append(str(prim_to_ctr[l][0])+SPDF[l].lower())
        nctrs.append(str(prim_to_ctr[l][1])+SPDF[l].lower())
    res.append('#BASIS SET: (%s) -> [%s]' % (','.join(nprims), ','.join(nctrs)))

    # pass 2: basis data
    for bas in basis:
        res.append('%-2s    %s' % (symb, SPDF[bas[0]]))
        for dat in bas[1:]:
            res.append(' '.join('%15.9f'%x for x in dat))
    return '\n'.join(res)

def convert_ecp_to_nwchem(symb, ecp):
    '''Convert the internal ecp format to NWChem format string'''
    from pyscf.gto.mole import _std_symbol
    symb = _std_symbol(symb)
    res = ['%-2s nelec %d' % (symb, ecp[0])]

    for ecp_block in ecp[1]:
        l = ecp_block[0]
        if l == -1:
            res.append('%-2s ul' % symb)
        else:
            res.append('%-2s %s' % (symb, SPDF[l].lower()))
        for r_order, dat in enumerate(ecp_block[1]):
            for e,c in dat:
                res.append('%d    %15.9f  %15.9f' % (r_order, e, c))
    return '\n'.join(res)

def _parse(raw_basis, optimize=False):
    basis_add = []
    for line in raw_basis:
        dat = line.strip()
        if not dat or dat.startswith('#'):
            continue
        elif dat[0].isalpha():
            key = dat.split()[1]
            if key == 'SP':
                basis_add.append([0])
                basis_add.append([1])
            else:
                basis_add.append([MAPSPDF[key]])
        else:
            try:
                line = [float(x) for x in dat.replace('D','e').split()]
            except BaseException as e:
                raise RuntimeError('\n' + e.message +
                                   '\nor the required basis file not existed.')
            if key == 'SP':
                basis_add[-2].append([line[0], line[1]])
                basis_add[-1].append([line[0], line[2]])
            else:
                basis_add[-1].append(line)
    basis_sorted = []
    for l in range(MAXL):
        basis_sorted.extend([b for b in basis_add if b[0] == l])

    if optimize:
        basis_sorted = optimize_contraction(basis_sorted)

    return basis_sorted

def optimize_contraction(basis):
    '''
    Optimize contraction: segment contraction -> general contraction
    '''
    bas_l = [[] for l in range(MAXL)]
    for b in basis:
        l = b[0]
        ec = numpy.array(b[1:]).T
        es = ec[0]
        cs = [c for c in ec[1:]]
        if bas_l[l]:
            for e_cs in bas_l[l]:
                if numpy.allclose(e_cs[0], es):
                    e_cs.extend(cs)
                    break
            else:
                bas_l[l].append([es] + cs)
        else:
            bas_l[l].append([es] + cs)

    basis = []
    for l, b in enumerate(bas_l):
        for e_cs in b:
            b_l = [l] + numpy.array(e_cs).T.tolist()
            basis.append(b_l)
    return basis

def _parse_ecp(raw_ecp):
    ecp_add = []
    nelec = None
    for line in raw_ecp:
        dat = line.strip()
        if not dat or dat.startswith('#'): # comment line
            continue
        elif dat[0].isalpha():
            key = dat.split()[1]
            if key == 'NELEC':
                nelec = int(dat.split()[2])
                continue
            elif key == 'UL':
                ecp_add.append([-1])
            else:
                ecp_add.append([MAPSPDF[key]])
            by_ang = [[], [], [], []]
            ecp_add[-1].append(by_ang)
        else:
            line = dat.replace('D','e').split()
            l = int(line[0])
            by_ang[l].append([float(x) for x in line[1:]])
    if nelec is not None:
        bsort = []
        for l in range(-1, MAXL):
            bsort.extend([b for b in ecp_add if b[0] == l])
        return [nelec, bsort]

if __name__ == '__main__':
    from pyscf import gto
    mol = gto.M(atom='O', basis='6-31g')
    print(load_ecp('lanl2dz.dat', 'Na'))
    b = load('ano.dat', 'Na')
    print(convert_basis_to_nwchem('Na', b))
    b = load_ecp('lanl2dz.dat', 'Na')
    print(convert_ecp_to_nwchem('Na', b))
