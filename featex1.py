import numpy as np
import re
from collections import Counter
import math

def AAC(fastas, **kw):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    header = []
    for i in AA:
        header.append(i)
    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        count = Counter(sequence)
        for key in count:
            count[key] = count[key]/len(sequence)
        code = []
        for aa in AA:
            code.append(count[aa])
        encodings.append(code)
    return np.array(encodings, dtype=float), header

def APAAC(fastas, lambdaValue=1, w=0.05, **kw):
    # ใช้แค่ 2 property
    records = []
    records.append("#   A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V")
    records.append("Hydrophobicity  0.62    -2.53   -0.78   -0.9    0.29    -0.85   -0.74   0.48    -0.4    1.38    1.06    -1.5    0.64    1.19    0.12    -0.18   -0.05   0.81    0.26    1.08")
    records.append("Hydrophilicity  -0.5    3   0.2 3   -1  0.2 3   0   -0.5    -1.8    -1.8    3   -1.3    -2.5    0   0.3 -0.4    -3.4    -2.3    -1.5")
    # *** SideChainMass ไม่ถูกใช้ (ให้เป็น APAAC 22 ฟีเจอร์) ***
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
            # เติม 0 ถ้าสั้นเกิน
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

def DPC(fastas, gap, **kw):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
    header = [] + diPeptides

    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        if len(sequence) < gap+2:
            code = [0.0] * 400
        else:
            code = []
            tmpCode = [0] * 400
            for j in range(len(sequence) - 2 + 1 - gap):
                tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j+gap+1]]] += 1
            if sum(tmpCode) != 0:
                tmpCode = [i/sum(tmpCode) for i in tmpCode]
            code = code + tmpCode
        encodings.append(code)
    return np.array(encodings, dtype=float), header

def Rvalue(aa1, aa2, AADict, Matrix):
    return sum([(Matrix[i][AADict[aa1]] - Matrix[i][AADict[aa2]]) ** 2 for i in range(len(Matrix))]) / len(Matrix)

def PAAC(fastas, lambdaValue=1, w=0.05, **kw):
    records = []
    records.append("#   A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V")
    records.append("Hydrophobicity  0.62    -2.53   -0.78   -0.9    0.29    -0.85   -0.74   0.48    -0.4    1.38    1.06    -1.5    0.64    1.19    0.12    -0.18   -0.05   0.81    0.26    1.08")
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
    for aa in AA:
        header.append('Xc1.' + aa)
    for n in range(1, lambdaValue + 1):
        header.append('Xc2.lambda' + str(n))

    for i in fastas:
        name, sequence= i[0], re.sub('-', '', i[1])
        if len(sequence) < lambdaValue + 1:
            code = [0.0] * (20 + lambdaValue)
        else:
            code = []
            theta = []
            for n in range(1, lambdaValue + 1):
                theta.append(
                    sum([Rvalue(sequence[j], sequence[j + n], AADict, AAProperty1) for j in range(len(sequence) - n)]) / (
                        len(sequence) - n))
            myDict = {}
            for aa in AA:
                myDict[aa] = sequence.count(aa)
            code = code + [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
            code = code + [(w * j) / (1 + w * sum(theta)) for j in theta]
        encodings.append(code)
    return np.array(encodings, dtype=float), header

def featex(fasta):
    # ตรวจสอบความยาวขั้นต่ำ
    # (ทุกตัวอย่างต้องยาว >= 2 สำหรับ DPC, PAAC, APAAC ที่ lambdaValue=1)
    # หากไม่ถึง เติมศูนย์ฟีเจอร์อัตโนมัติในแต่ละฟังก์ชันอยู่แล้ว
    fname = []
    feat0 = AAC(fasta)[0]; fname.append('AAC')
    feat1 = DPC(fasta, 0)[0]; fname.append('DPC')
    feat2 = PAAC(fasta, 1)[0]; fname.append('PAAC')
    feat3 = APAAC(fasta, 1)[0]; fname.append('APAAC')

    allfeat_pos = np.hstack((feat0, feat1, feat2, feat3))
    numdesc = len(fname)
    f = []
    before = 0
    for i in range(numdesc):
        after = before + eval('feat%d.shape[1]' % (i))
        f.append(list(range(before, after)))
        before = after
    return allfeat_pos, f, fname
