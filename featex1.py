# featex1.py
import numpy as np
import math
import re
from collections import Counter

# ---- 1. AAC ----
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

# ---- 2. APAAC ----
def APAAC(fastas, lambdaValue=1, w=0.05, **kw):
    records = [
        "#   A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V",
        "Hydrophobicity  0.62    -2.53   -0.78   -0.9    0.29    -0.85   -0.74   0.48    -0.4    1.38    1.06    -1.5    0.64    1.19    0.12    -0.18   -0.05   0.81    0.26    1.08",
        "Hydrophilicity  -0.5    3   0.2 3   -1  0.2 3   0   -0.5    -1.8    -1.8    3   -1.3    -2.5    0   0.3 -0.4    -3.4    -2.3    -1.5",
        "SideChainMass   15  101 58  59  47  72  73  1   82  57  57  73  75  91  42  31  45  130 107 43"
    ]
    AA = ''.join(records[0].rstrip().split()[1:])
    AADict = {AA[i]: i for i in range(len(AA))}
    AAProperty, AAPropertyNames = [], []
    for i in range(1, len(records)):
        array = records[i].rstrip().split()
        AAProperty.append([float(j) for j in array[1:]])
        AAPropertyNames.append(array[0])
    # Normalize
    AAProperty1 = []
    for i in AAProperty:
        meanI = sum(i) / 20
        fenmu = math.sqrt(sum([(j - meanI) ** 2 for j in i]) / 20)
        AAProperty1.append([(j - meanI) / fenmu for j in i])
    encodings, header = [], []
    for i in AA:
        header.append('Pc1.' + i)
    for j in range(1, lambdaValue + 1):
        for i in AAPropertyNames:
            header.append('Pc2.' + i + '.' + str(j))
    for name, seq in fastas:
        sequence = re.sub('-', '', seq)
        code = []
        theta = []
        for n in range(1, lambdaValue + 1):
            for j in range(len(AAProperty1)):
                thetasum = 0
                for k in range(len(sequence) - n):
                    thetasum += AAProperty1[j][AADict[sequence[k]]] * AAProperty1[j][AADict[sequence[k + n]]]
                theta.append(thetasum / (len(sequence) - n))
        myDict = {aa: sequence.count(aa) for aa in AA}
        code += [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
        code += [w * value / (1 + w * sum(theta)) for value in theta]
        encodings.append(code)
    return np.array(encodings, dtype=float), header

# ---- 3. DPC ----
def DPC(fastas, gap=0, **kw):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
    AADict = {AA[i]: i for i in range(len(AA))}
    encodings = []
    for name, seq in fastas:
        sequence = re.sub('-', '', seq)
        code = [0] * 400
        for j in range(len(sequence) - 1 - gap):
            first, second = sequence[j], sequence[j+gap+1]
            code[AADict[first]*20 + AADict[second]] += 1
        total = sum(code)
        if total > 0:
            code = [i / total for i in code]
        encodings.append(code)
    return np.array(encodings, dtype=float), diPeptides

# ---- 4. PAAC ----
def Rvalue(aa1, aa2, AADict, Matrix):
    return sum([(Matrix[i][AADict[aa1]] - Matrix[i][AADict[aa2]]) ** 2 for i in range(len(Matrix))]) / len(Matrix)

def PAAC(fastas, lambdaValue=1, w=0.05, **kw):
    records = [
        "#   A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V",
        "Hydrophobicity  0.62    -2.53   -0.78   -0.9    0.29    -0.85   -0.74   0.48    -0.4    1.38    1.06    -1.5    0.64    1.19    0.12    -0.18   -0.05   0.81    0.26    1.08",
        "Hydrophilicity  -0.5    3   0.2 3   -1  0.2 3   0   -0.5    -1.8    -1.8    3   -1.3    -2.5    0   0.3 -0.4    -3.4    -2.3    -1.5",
        "SideChainMass   15  101 58  59  47  72  73  1   82  57  57  73  75  91  42  31  45  130 107 43"
    ]
    AA = ''.join(records[0].rstrip().split()[1:])
    AADict = {AA[i]: i for i in range(len(AA))}
    AAProperty, AAPropertyNames = [], []
    for i in range(1, len(records)):
        array = records[i].rstrip().split()
        AAProperty.append([float(j) for j in array[1:]])
        AAPropertyNames.append(array[0])
    # Normalize
    AAProperty1 = []
    for i in AAProperty:
        meanI = sum(i) / 20
        fenmu = math.sqrt(sum([(j - meanI) ** 2 for j in i]) / 20)
        AAProperty1.append([(j - meanI) / fenmu for j in i])
    encodings, header = [], []
    for aa in AA:
        header.append('Xc1.' + aa)
    for n in range(1, lambdaValue + 1):
        header.append('Xc2.lambda' + str(n))
    for name, seq in fastas:
        sequence = re.sub('-', '', seq)
        code = []
        theta = []
        for n in range(1, lambdaValue + 1):
            theta.append(
                sum([Rvalue(sequence[j], sequence[j + n], AADict, AAProperty1) for j in range(len(sequence) - n)]) / (len(sequence) - n)
            )
        myDict = {aa: sequence.count(aa) for aa in AA}
        code += [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
        code += [(w * j) / (1 + w * sum(theta)) for j in theta]
        encodings.append(code)
    return np.array(encodings, dtype=float), header

# ---- 5. รวมฟีเจอร์ ----
def featex(fasta):
    feat0, _ = AAC(fasta)
    feat1, _ = DPC(fasta, 0)
    feat2, _ = PAAC(fasta, 1)
    feat3, _ = APAAC(fasta, 1)
    allfeat_pos = np.hstack((feat0, feat1, feat2, feat3))
    return allfeat_pos
