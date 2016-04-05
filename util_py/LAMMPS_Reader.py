import re
import sys
class LAMMPS_Reader:
    def __init__(self, state=None, nonbondFix=None, bondFix=None, angleFix=None, dihedralFix=None, improperFix=None, unitLen = 0, unitEng = 0, unitMass = 0):
        assert(state != None)
        assert(len(inputFns))
        self.state = state
        self.nonbondFix = nonbondFix
        self.bondFix = bondFix
        self.angleFix = angleFix
        self.dihedralFix = dihedralDix
        self.improperFix = improperFix
        assert(unitLen != 0 and unitEng != 0 and unitMass != 0)
        self.unitLen = unitLen
        self.unitEng = unitEng
        self.unitMass = unitMass

    def read(self, dataFn='', inputFns=[]):
        assert(len(inputFns))
        self.dataFile = open(dataFN, 'r')
        self.inputFiles = [open(inputFn, 'r') for inputFn in inputFns]
        self.allFiles = [dataFile ] + inputFiles
        self.allFileLines = [f.readlines() for f in self.allFiles]

        setAtomTypes()
    def isNums(self, bits):
        for b in bits:
            try:
                float(b)
            except ValueError:
                return False
        return True

    def setAtomTypes(self):

        masses = readSection(self.dataFile, 'Mass')
        for pair in masses:
            typeName = pair[0]
            mass = float(pair[1]) / self.unitMass
            state.addSpecies(typeName, mass)


    def readSection(self, dataFile, header):
        readData = []
        lineIdx = 0
        while lineIdx < len(dataFile):
            if header.search(dataFile[lineIdx]):
                break
            lineIdx+=1
        while not len(dataFile[lineIdx].split()):
            lineIdx+=1
        while len(dataFile[lineIdx]):
            bits = dataFile[lineIdx].split()
            if isNums(bits):
                readData.append(bits)
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
            int lineNum = 0
            while numOccur < num and lineNum < len(f):
                if regex.search(f[lineNum]):
                    res.append(f[lineNum])
                lineNum+=1
            fIdx+=1
        return res

