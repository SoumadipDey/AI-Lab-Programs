import numpy as np
membersA = input("Enter the members of the set A: ").split()
deg_memA = list(map(float,input("Enter the degree of membership for A Set: ").split()))

membersB = input("Enter the members of the set B: ").split()
deg_memB = list(map(float,input("Enter the degree of membership for B Set: ").split()))

setA = {}
setB = {}

for memA,degA in zip(membersA, deg_memA):
    setA[memA] = degA

for memB,degB in zip(membersB, deg_memB):
    setB[memB] = degB

print("Set A:\n",setA)
print("\nSet B:\n",setB)

def fuzzyUnion(setA:dict, setB:dict):
    setAUB = {}
    all_members = set(setA.keys()).union(setB.keys())
    for mem in all_members:
        setAUB[mem] = max(setA.get(mem,0.0), setB.get(mem,0.0))
    return setAUB

def fuzzyIntersection(setA:dict, setB:dict):
    setAandB = {}
    all_members = set(setA.keys()).union(setB.keys())
    for mem in all_members:
        setAandB[mem] = min(setA.get(mem,0.0), setB.get(mem,0.0))
    return setAandB

def fuzzyComplement(setA:dict):
    setAcomplement = {mem: round(1.0 - setA[mem],2) for mem in setA.keys()}
    return setAcomplement

def fuzzyAlgebraicSum(setA:dict, setB:dict):
    setAplusB = {}
    all_members = set(setA.keys()).union(setB.keys())
    for mem in all_members:
        setAplusB[mem] = round(
            setA.get(mem,0.0) + 
            setB.get(mem,0.0) - 
            setA.get(mem,0.0) * setB.get(mem,0.0),2)
    return setAplusB

def fuzzyBoundedSum(setA:dict, setB:dict):
    setAplusB = {}
    all_members = set(setA.keys()).union(setB.keys())
    for mem in all_members:
        setAplusB[mem] = min(1.0, round(setA.get(mem,0.0) + setB.get(mem,0.0))) 
    return setAplusB

def fuzzyAlgebraicProduct(setA:dict, setB:dict):
    setAdotB = {}
    all_members = set(setA.keys()).union(setB.keys())
    for mem in all_members:
            setAdotB[mem] = round(setA.get(mem,0.0) * setB.get(mem,0.0),2)
    return setAdotB

def fuzzyCartesianProduct(setA:dict, setB:dict):
    cp_matrix = np.zeros((len(setA),len(setB)))
    for i, memA in enumerate(setA.keys()):
        for j, memB in enumerate(setB.keys()):
            cp_matrix[i][j] = min(setA[memA],setB[memB])

    return cp_matrix

print("\nA Union B:\n",fuzzyUnion(setA,setB))
print("\nA Intersection B:\n",fuzzyIntersection(setA,setB))
print("\nA Complement:\n",fuzzyComplement(setA))
print("\nB Complement:\n",fuzzyComplement(setB))
print("\nAlgebriac Sum of A & B:\n",fuzzyAlgebraicSum(setA,setB))
print("\nBounded Sum of A & B:\n",fuzzyBoundedSum(setA,setB))
print("\nAlgebriac Product of A & B:\n",fuzzyAlgebraicProduct(setA,setB))
print("\nCartesian Product of Fuzzy Sets A and B: \n",fuzzyCartesianProduct(setA, setB))