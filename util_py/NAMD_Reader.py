import re
import sys
import math
DEGREES_TO_RADIANS = math.pi / 180.


class NAMD_Bonded_Forcer:
    def __init__(self, type, atomTypes):
        self.type = type
#these are types with the atomTypePrefix
        self.atomTypes = atomTypes
    def matchLevel(self, atoms):
#0 -> no match, 1-> total wildcard, 2->atom + X, 3->perfect
        matchLevel = 3
        for i in range(len(self.atomTypes)):
            atomType = self.atomTypes[i]
            if atomType == 'x':
                if matchLevel > 1:
                    matchLevel = 1
            elif atomType[-1] == 'X':
                if atomType[:-1] == atoms[i].type[:len(atomType)-1]:
                    if matchLevel > 2:
                        matchLevel = 2
            elif atomType != atoms[i].type:
                matchLevel = 0
        return matchLevel

#set to wildcard



class NAMD_Reader:
    def __init__(self, state=None, nonbondFix=None, bondFix=None, angleFix=None, dihedralFix=None, improperFix=None, unitLen = 0, unitEng = 0, unitMass = 0, atomTypePrefix = '', setBounds=True):
        assert(state != None)
        self.state = state
        self.nonbondFix = nonbondFix
        self.bondFix = bondFix
        self.angleFix = angleFix
        self.dihedralFix = dihedralFix
        self.improperFix = improperFix
        assert(unitLen != 0 and unitEng != 0 and unitMass != 0)
        self.unitLen = unitLen
        self.unitEng = unitEng
        self.unitMass = unitMass
        self.myAtomTypeIds = []
        self.myAtomHandles = []
        self.atomTypePrefix = atomTypePrefix
        self.setBounds = setBounds

        self.NAMDBondTypes = []
        self.NAMDAngleTypes = []
        self.NAMDDihedralTypes = []
        self.NAMDImproperTypes = []

        self.namdToSimId = {}

    def read(self, inputFn='', structureFn='', coordinatesFn='', parametersFn=''):
        self.inputFile = open(inputFn, 'r')
        self.structureFile = open(structureFn, 'r')
        self.structureFileLines = self.structureFile.readlines()
        self.coordinateFile = open(coordinatesFn, 'r')
        self.coordinateFileLines = self.coordinateFile.readlines()
        self.parameterFile = open(parametersFn, 'r')
        self.parameterFileLines = self.parameterFile.readlines()


        self.readAtoms()
        print len(self.state.atoms)
        #self.readAtomTypes()


        #might not want to change bounds if you just adding parameters to an existing simulation
        if self.setBounds:
            self.readBounds()
        if self.nonbondFix:
            self.readPairCoefs()
        '''
        if self.bondFix != None:
            self.readBonds()
            self.readBondCoefs()
        if self.angleFix != None:
            self.readAngles()
            self.readAngleCoefs()
        if self.dihedralFix != None:
            self.readDihedrals()
            self.readDihedralCoefs()
        if self.improperFix != None:
            self.readImpropers()
            self.readImproperCoefs()
            '''




    def readAtoms(self):
        nAtoms = 0
        idxStart = 0
        for i in xrange(len(self.structureFileLines)):
            if 'NATOM' in self.structureFileLines[i]:
                nAtoms = int(self.structureFileLines[i].split()[0])
                idxStart = i+1
                break
        for i in range(idxStart, idxStart + nAtoms):
            bits = self.structureFileLines[i].split()
            atomType = self.atomTypePrefix + bits[5]
            mass = float(bits[7])
            q = float(bits[6])
            self.state.atomParams.addSpecies(atomType, mass)
            atomId = self.state.addAtom(atomType, self.state.Vector(), q)
            namdId = int(bits[0])
            self.namdToSimId[namdId] = atomId
        for i in range(1, nAtoms+1):
#deposit.rcsb.org/adit/docs/pdb_atom_format.html
            line = self.coordinateFileLines[i]
            self.state.atoms[i-1].pos[0] = float(line[31:39]) / self.unitLen
            self.state.atoms[i-1].pos[1] = float(line[39:47]) / self.unitLen
            self.state.atoms[i-1].pos[2] = float(line[47:55]) / self.unitLen


    def readPairCoefs(self):
        for i in range(len(self.parameterFileLines)):
            if 'NONBONDED' in self.stripComments(self.parameterFileLines[i]):
                break
        i+=2

        while i < len(self.parameterFileLines) and self.parameterFileLines[i] != '':
            bits = self.stripComments(self.parameterFileLines[i]).split()
            if len(bits):
                print bits
                handle = self.atomTypePrefix + bits[0]
                epsInput = float(bits[2])
                rMinInput = float(bits[3])
                eps = -epsInput / self.unitEng
                sigma = (rMinInput * 2) / pow(2.0, 1.0 / 6.0) / self.unitLen#IS THE X2 CORRECT?  ASK AMIN
                self.nonbondFix.setParameter('sig', handle, handle, sigma)
                self.nonbondFix.setParameter('eps', handle, handle, sigma)
            i+=1





    def isNums(self, bits):
        for b in bits:
            if b[0] == '#':
                break
            try:
                float(b)
            except ValueError:
                return False
        return len(bits)
    def stripComments(self, line):
        if '!' in line:
            return line[:line.index('!')]
        return line
    def emptyLine(self, line):
        bits = line.split()
        return len(bits)==0 or bits[0][0]=='!'
    def emptyLineSplit(self, bits):
        return len(bits)==0 or bits[0][0]=='!'


