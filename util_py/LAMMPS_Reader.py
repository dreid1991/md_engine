import re
import sys
class LAMMPS_Reader:
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

        self.LMPTypeToSimTypeBond = {}
        self.LMPTypeToSimTypeAngle = {}
        self.LMPTypeToSimTypeDihedral = {}
        self.LMPTypeToSimTypeImproper = {}

    def read(self, dataFn='', inputFns=[]):
        self.dataFile = open(dataFn, 'r')
        self.inputFiles = [open(inputFn, 'r') for inputFn in inputFns]
        self.allFiles = [self.dataFile ] + self.inputFiles
        self.dataFileLines = self.dataFile.readlines()
        self.inFileLines = [f.readlines() for f in self.inputFiles]
        self.allFileLines = [self.dataFileLines] + self.inFileLines
        self.isMolecular = len(self.readSection(self.dataFileLines, re.compile('Bonds'))) #this is slow, should write something to test if section exists

        self.readAtomTypes()
        self.atomIdToIdx = {}


        #might not want to change bounds if you just adding parameters to an existing simulation
        if self.setBounds:
            self.readBounds()
        self.readAtoms()
        if self.nonbondFix:
            self.readPairCoefs() #implement this
        for a in self.state.atoms:
            print a.pos
        if self.bondFix != None:
            self.readBonds()
            self.readBondCoefs()
        if self.angleFix != None:
            self.readAngles() #todo
            self.readAngleCoefs()
        if self.dihedralFix != None:
            self.readDihedrals()
            self.readDihedralCoefs()
        if self.improperFix != None:
            self.readImpropers()
            self.readImproperCoefs()
        #now read parameters
    def isNums(self, bits):
        for b in bits:
            if b[0] == '#':
                break
            try:
                float(b)
            except ValueError:
                return False
        return len(bits)
    def emptyLine(self, line):
        bits = line.split()
        return len(bits)==0 or bits[0][0]=='#'

    def readAtomTypes(self):
        numAtomTypesRE = re.compile('^[\s][\d]+[\s]+atom[\s]+types')
        numTypesLines = self.scanFilesForOccurance(numAtomTypesRE, [self.dataFileLines])
        assert(len(numTypesLines) == 1)

        numTypes = int(numTypesLines[0].split()[0])
#adding atoms with mass not set
        for i in range(numTypes):
            self.myAtomHandles.append(str(self.atomTypePrefix) + str(i))
            self.myAtomTypeIds.append(self.state.atomParams.addSpecies(self.myAtomHandles[-1], -1))

#now getting / setting masses
        masses = self.readSection(self.dataFileLines, re.compile('Mass'))
        for i, pair in enumerate(masses):
            typeIdx = self.myAtomTypeIds[i]
            mass = float(pair[1]) / self.unitMass
            self.state.atomParams.masses[typeIdx] = mass


    def readBounds(self):
        #reBase = '^\s+[\-\.\d]+\s+[\-\.\d]\s+%s\s+%s\s$'
        reBase = '^\s+[\-\.\d]+[\s]+[\-\.\d]+[\s]+%s[\s]+%s'
        bits = [('xlo', 'xhi'), ('ylo', 'yhi'), ('zlo', 'zhi')]
        lo = self.state.Vector()
        hi = self.state.Vector()
        for i, bit in enumerate(bits):
            dimRe = re.compile(reBase % bit)
            lines = self.scanFilesForOccurance(dimRe, [self.dataFileLines])
            assert(len(lines) == 1)
            lineSplit = lines[0].split()
            lo[i] = float(lineSplit[0]) / self.unitLen
            hi[i] = float(lineSplit[1]) / self.unitLen
        self.state.bounds.lo = lo
        self.state.bounds.hi = hi
#code SHOULD be in place to let one just change lo, hi like this.  please make sure
    def readAtoms(self):
        raw = self.readSection(self.dataFileLines, re.compile('Atoms'))
        areCharges =  (len(raw[0]) == 7) if self.isMolecular else (len(raw[0]) == 6)
        for atomLine in raw:
            pos = self.state.Vector()
            pos[0] = float(atomLine[-3]) / self.unitLen
            pos[1] = float(atomLine[-2]) / self.unitLen
            pos[2] = float(atomLine[-1]) / self.unitLen
            atomType = -1
            charge = 0
            if areCharges:
                charge = float(atomLine[-4])
                atomType = int(atomLine[-5])
            else:
                atomType = int(atomLine[-4])
            handle = self.myAtomHandles[atomType-1] #b/c lammps starts at 1
            self.atomIdToIdx[int(atomLine[0])] = len(self.state.atoms)
            self.state.addAtom(handle = handle, pos = pos, q = charge)
    def readBonds(self):
        raw = self.readSection(self.dataFileLines, re.compile('Bonds'))
        currentTypes = self.bondFix.getTypeIds()
        if len(currentTypes):
            typeOffset = max(currentTypes) + 1
        else:
            typeOffset = 0
        for bondLine in raw:
            bondType = int(bondLine[1])
            idA = int(bondLine[2])
            idB = int(bondLine[3])
            idxA = self.atomIdToIdx[idA]
            idxB = self.atomIdToIdx[idB]
            simType = typeOffset + bondType
            self.bondFix.createBond(self.state.atoms[idxA], self.state.atoms[idxB], type=simType)

            self.LMPTypeToSimTypeBond[bondType] = simType

    def readAngles(self):
        raw = self.readSection(self.dataFileLines, re.compile('Angles'))
        currentTypes = self.angleFix.getTypeIds()
        if len(currentTypes):
            typeOffset = max(currentTypes) + 1
        else:
            typeOffset = 0
        for line in raw:
            type = int(bondLine[1])
            ids = [int(x) for a in [line[2], line[3], line[4]]]
            idxs = [self.atomIdToIdx[id] for id in ids]
            simType = typeOffset + type
            self.angleFix.createAngle(self.state.atoms[idxs[0]], self.state.atoms[idxs[1]], self.state.atoms[idxs[2]], type=simType)

            self.LMPTypeToSimTypeAngle[type] = simType

    def readDihedrals(self):
        raw = self.readSection(self.dataFileLines, re.compile('Dihedrals'))
        currentTypes = self.dihedralFix.getTypeIds()
        if len(currentTypes):
            typeOffset = max(currentTypes) + 1
        else:
            typeOffset = 0
        for line in raw:
            type = int(bondLine[1])
            ids = [int(x) for a in [line[2], line[3], line[4]]]
            idxs = [self.atomIdToIdx[id] for id in ids]
            simType = typeOffset + type
            self.angleFix.createAngle(self.state.atoms[idxs[0]], self.state.atoms[idxs[1]], self.state.atoms[idxs[2]], self.state.atoms[idxs[3]], type=simType)

            self.LMPTypeToSimTypeDihedral[type] = simType
    def readImpropers(self):
        raw = self.readSection(self.dataFileLines, re.compile('Impropers'))
        currentTypes = self.improperFix.getTypeIds()
        if len(currentTypes):
            typeOffset = max(currentTypes) + 1
        else:
            typeOffset = 0
        for line in raw:
            type = int(bondLine[1])
            ids = [int(x) for a in [line[2], line[3], line[4]]]
            idxs = [self.atomIdToIdx[id] for id in ids]
            simType = typeOffset + type
            self.angleFix.createAngle(self.state.atoms[idxs[0]], self.state.atoms[idxs[1]], self.state.atoms[idxs[2]], self.state.atoms[idxs[3]], type=simType)

            self.LMPTypeToSimTypeImproper[type] = simType

    def readPairCoefs(self):
        rawData = self.readSection(self.dataFileLines, re.compile('Pair Coeffs'))
        for line in rawData:
#            type = self.LMPTypeToSimTypeBond[int(line[0])]
#            rest = [float(x) for x in line[1:]]
#            self.bondFix.setBondTypeCoefs(type, *rest)
#FINISH THIS

    def readBondTypes(self):
        rawData = self.readSection(self.dataFileLines, re.compile('Bond Coeffs'))
        for line in rawData:
            type = self.LMPTypeToSimTypeBond[int(line[0])]
            rest = [float(x) for x in line[1:]]
            self.bondFix.setBondTypeCoefs(type, *rest)
        rawInput = self.scanFilesForOccurance(re.compile('bond coeff[\s\d\-\.]+'), self.inFileLine, num=-1)
        for line in rawInput:
            type = self.LMPTypeToSimTypeBond[int(line[1])]
            rest = [float(x) for x in line[2:]]
            self.bondFix.setBondTypeCoefs(type, *rest)

    def readAngleTypes(self):
        rawData = self.readSection(self.dataFileLines, re.compile('Angle Coeffs'))
        for line in rawData:
            type = self.LMPTypeToSimTypeAngle[int(line[0])]
            rest = [float(x) for x in line[1:]]
            self.angleFix.setAngleTypeCoefs(type, *rest)
        rawInput = self.scanFilesForOccurance(re.compile('angle coeff[\s\d\-\.]+'), self.inFileLine, num=-1)
        for line in rawInput:
            type = self.LMPTypeToSimTypeAngle[int(line[1])]
            rest = [float(x) for x in line[2:]]
        self.angleFix.setAngleTypesCoefs(type, *rest)



    def readDihedralTypes(self):
        rawData = self.readSection(self.dataFileLines, re.compile('Dihedral Coeffs'))
        for line in rawData:
            type = self.LMPTypeToSimTypeDihedral[int(line[0])]
            rest = [float(x) for x in line[1:]]
            self.dihedralFix.setDihedralTypeCoefs(type, *rest)
        rawInput = self.scanFilesForOccurance(re.compile('dihedral coeff[\s\d\-\.]+'), self.inFileLine, num=-1)
        for line in rawInput:
            type = self.LMPTypeToSimTypeDihedral[int(line[1])]
            rest = [float(x) for x in line[2:]]
            self.dihedralFix.setDihedralTypesCoefs(type, *rest)



    def readImproperTypes(self):
        rawData = self.readSection(self.dataFileLines, re.compile('Improper Coeffs'))
        for line in rawData:
            type = self.LMPTypeToSimTypeImproper[int(line[0])]
            rest = [float(x) for x in line[1:]]
            self.improperFix.setImproperTypeCoefs(type, *rest)
        rawInput = self.scanFilesForOccurance(re.compile('improper coeff[\s\d\-\.]+'), self.inFileLine, num=-1)
        for line in rawInput:
            type = self.LMPTypeToSimTypeImproper[int(line[1])]
            rest = [float(x) for x in line[2:]]
            self.improperFix.setImproperTypesCoefs(type, *rest)


    def stripComments(self, line):
        if '#' in line:
            return line[:line.index('#')]
        else:
            return line

    def readSection(self, dataFileLines, header):
        readData = []
        lineIdx = 0
        while lineIdx < len(dataFileLines):
            if header.search(dataFileLines[lineIdx]):
                lineIdx+=1
                break

            lineIdx+=1
        while lineIdx < len(dataFileLines) and self.emptyLine(dataFileLines[lineIdx]):
            lineIdx+=1
        while lineIdx < len(dataFileLines) and len(dataFileLines[lineIdx]):
            line = self.stripComments(dataFileLines[lineIdx])
            bits = line.split()
            if self.isNums(bits):
                readData.append(bits)
                lineIdx+=1
            else:
                break
        return readData

    def scanFilesForOccurance(self, regex, files, num=1):
        numOccur = 0
        fIdx = 0
        res = []
        if num==-1:
            num = sys.maxint
        while numOccur < num and fIdx < len(files):
            f = files[fIdx]
            lineNum = 0
            while numOccur < num and lineNum < len(f):
                if regex.search(f[lineNum]):
                    res.append(f[lineNum])
                lineNum+=1
            fIdx+=1
        return res

