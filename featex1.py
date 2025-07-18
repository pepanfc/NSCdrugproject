import numpy as np
import math
import re
from collections import Counter

def AAC(fastas, **kw):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    header = list(AA)
    for name, seq in fastas:
        sequence = re.sub('-', '', seq)
        count = Counter(sequence)
        total = len(sequence)
        code = [count.get(aa, 0) / total if total > 0 else 0 for aa in AA]
        encodings.append(code)
    return np.array(encodings, dtype=float), header

def APAAC(fastas, lambdaValue=1, w=0.05, **kw):
    records = []
    records.append("#   A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V")
    # ใช้เฉพาะ 2 property ที่ต้องการ
    records.append("Hydrophilicity  -0.5    3   0.2 3   -1  0.2 3   0   -0.5    -1.8    -1.8    3   -1.3    -2.5    0   0.3 -0.4    -3.4    -2.3    -1.5")
    records.append("SideChainMass   15  101 58  59  47  72  73  1   82  57  57  73  75  91  42  31  45  130 107 43")

    AA = ''.join(records[0].rstrip().split()[1:])
    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i
    AAProperty = []
    AAPropertyNames = []
    for i in range(1, len(records)):
        array = records[i].rstrip().split() if records[i].rstrip() != '' else None
        AAProperty.append([float(j) for j in array[1:]])
        AAPropertyNames.append(array[0])

    AAProperty1 = []
    for i in AAProperty:
        meanI = sum(i) / 20
        fenmu = math.sqrt(sum([(j - meanI) ** 2 for j in i]) / 20)
        AAProperty1.append([(j - meanI) / fenmu for j in i])

    encodings = []
    header = []
    for i in AA:
        header.append('Pc1.' + i)
    for j in range(1, lambdaValue + 1):
        for i in AAPropertyNames:
            header.append('Pc2.' + i + '.' + str(j))

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        if len(sequence) < lambdaValue + 1:
            code = [0.0] * (20 + len(AAPropertyNames)*lambdaValue)
        else:
            code = []
            theta = []
            for n in range(1, lambdaValue + 1):
                for j in range(len(AAProperty1)):
                    theta.append(sum([AAProperty1[j][AADict[sequence[k]]] * AAProperty1[j][AADict[sequence[k + n]]] for k in
                                      range(len(sequence) - n)]) / (len(sequence) - n))
            myDict = {}
            for aa in AA:
                myDict[aa] = sequence.count(aa)
            code = code + [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
            code = code + [w * value / (1 + w * sum(theta)) for value in theta]
        encodings.append(code)
    return np.array(encodings, dtype=float), header

def DPC(fastas, gap=0, **kw):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    AADict = {aa: i for i, aa in enumerate(AA)}
    diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
    encodings = []
    header = diPeptides
    for name, seq in fastas:
        sequence = re.sub('-', '', seq)
        L = len(sequence)
        tmpCode = [0] * 400
        for j in range(L - gap - 1):
            aa1 = sequence[j]
            aa2 = sequence[j + gap + 1]
            if aa1 in AADict and aa2 in AADict:
                idx = AADict[aa1] * 20 + AADict[aa2]
                tmpCode[idx] += 1
        total = sum(tmpCode)
        code = [i / total if total > 0 else 0 for i in tmpCode]
        encodings.append(code)
    return np.array(encodings, dtype=float), header

def Rvalue(aa1, aa2, AADict, Matrix):
    return sum([(Matrix[i][AADict.get(aa1, 0)] - Matrix[i][AADict.get(aa2, 0)]) ** 2 for i in range(len(Matrix))]) / len(Matrix)

def PAAC(fastas, lambdaValue=30, w=0.05, **kw):
    records = [
        "#   A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V",
        "Hydrophobicity  0.62 -2.53 -0.78 -0.9 0.29 -0.85 -0.74 0.48 -0.4 1.38 1.06 -1.5 0.64 1.19 0.12 -0.18 -0.05 0.81 0.26 1.08",
        "Hydrophilicity  -0.5 3 0.2 3 -1 0.2 3 0 -0.5 -1.8 -1.8 3 -1.3 -2.5 0 0.3 -0.4 -3.4 -2.3 -1.5",
        "SideChainMass   15 101 58 59 47 72 73 1 82 57 57 73 75 91 42 31 45 130 107 43"
    ]
    AA = ''.join(records[0].split()[1:])
    AADict = {aa: i for i, aa in enumerate(AA)}
    AAProperty = []
    AAPropertyNames = []
    for line in records[1:]:
        arr = line.rstrip().split()
        AAProperty.append([float(j) for j in arr[1:]])
        AAPropertyNames.append(arr[0])
    # Normalize
    AAProperty1 = []
    for arr in AAProperty:
        mean_val = sum(arr) / 20
        std = math.sqrt(sum([(j - mean_val) ** 2 for j in arr]) / 20)
        AAProperty1.append([(j - mean_val) / std for j in arr])
    encodings = []
    header = ['Xc1.' + aa for aa in AA]
    for n in range(1, lambdaValue + 1):
        header.append('Xc2.lambda' + str(n))
    for name, seq in fastas:
        sequence = re.sub('-', '', seq)
        L = len(sequence)
        code = []
        theta = []
        for n in range(1, lambdaValue + 1):
            if L - n == 0:
                theta.append(0.0)
                continue
            th = sum([Rvalue(sequence[j], sequence[j + n], AADict, AAProperty1)
                      for j in range(L - n)]) / (L - n)
            theta.append(th)
        myDict = Counter(sequence)
        code += [myDict.get(aa, 0) / (1 + w * sum(theta)) if (1 + w * sum(theta)) != 0 else 0 for aa in AA]
        code += [(w * t) / (1 + w * sum(theta)) if (1 + w * sum(theta)) != 0 else 0 for t in theta]
        encodings.append(code)
    return np.array(encodings, dtype=float), header
