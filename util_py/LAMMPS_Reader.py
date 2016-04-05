import re

class LAMMPS_Reader:
    def __init__(self,state=None, nonbondFix=None, bondFix=None, angleFix=None, dihedralFix=None, improperFix=None):
        assert(state != None)
        assert(len(inputFns))
        self.state = state
        self.nonbondFix = nonbondFix
        self.bondFix = bondFix
        self.angleFix = angleFix
        self.dihedralFix = dihedralDix
        self.improperFix = improperFix
    def read(dataFn='', inputFns=[]):
        assert(len(inputFns))
        self.dataFile = open(dataFN, 'r')
        self.inputFiles = [open(inputFn, 'r') for inputFn in inputFns]
        self.allFiles = [dataFile ] + inputFiles
        self.allFileLines = [f.readlines() for f in self.allFiles]

        setAtomTypes

    def scanFilesForOccurance(regex, files, num=1):
        numOccur = 0
        fIdx = 0
        res = []
        while numOccur < num:
            f = files[fIdx]
            int lineNum = 0
            while numOccur < num and lineNum < len(f):
                if regex.search(f[lineNum]):
                    res.append(f[lineNum])
                lineNum+=1
            fIdx+=1
        return res
