#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# y = x or x^2, Ver 0.6 2022/10/20

import argparse
import numpy as np
import matplotlib.pyplot as plt
import math

from qiskit import(IBMQ,
                   QuantumRegister,
                   QuantumCircuit,
                   transpile,
                   assemble,
                   execute,
                   BasicAer)
from qiskit.quantum_info.operators import Operator
from qiskit.extensions import UnitaryGate
from qiskit.circuit.add_control import add_control

if __name__ == '__main__':
    option = argparse.ArgumentParser()

    option.add_argument('-f', '--fcn', default='linear',
                        help='Function to be integrate')
    option.add_argument("-d", "--div", default=5, \
                        help="Log(2) of division num.", type=int)
    option.add_argument("-c", "--count", default=0, \
                        help="Num. of counting qubits.", type=int)
    option.add_argument("-b", "--backend", default='qasm_simulator', \
                        help="Provider")
    option.add_argument("-s", "--shots", default=2048, help="SHOT", type=int)
#    option.add_argument("-o", "--out_file", default=False, help="out file")
    args = option.parse_args()

# Select backend
    if(args.shots > 0):
        shots = args.shots
    else:
        shots = 2048

    if('ibmq' in args.backend):
        provider = IBMQ.load_account()
        provider.backends()
        q_back = provider.get_backend(args.backend)
    else:
        q_back = BasicAer.get_backend(args.backend)

    n = args.div # Num. of division = 2**n default = 5

    if(args.count):
        t = args.count # Num. of counting qubits
    else:
        t = n          # Num. of counting qubits

    b = math.pi * 0.5 # 積分区間
#    b = 1.             # for TEST
    alpha = 0.5 # 代表点の位置を決めるファクタ, 0.5　の場合は各区間の中央

    if(n==3):
        if(args.fcn=='linear'):
            theta0 = [0.33, 0.49, 0.75]
            theta1 = [-0.12, -0.13, -0.15]
        else:
            theta0 = [0.14, 0.26, 0.54]
            theta1 = [0.00, 0.02, 0.08]
    elif(n==4):
        if(args.fcn=='linear'):
            theta0 = [0.22, 0.33, 0.48, 0.75]
            theta1 = [-0.07, -0.08, -0.11, -0.07, -0.13, -0.14]
        else:
            theta0 = [0.07, 0.14, 0.27, 0.55]
            theta1 = [0.00, 0.00, 0.01, 0.01, 0.04, 0.08]
    else:
        if(args.fcn=='linear'):
            theta0 = [0.14, 0.21, 0.32, 0.48, 0.75]
            theta1 = [-0.03, -0.04, -0.06, -0.05, -0.08, -0.10, -0.07, -0.08, -0.08, -0.09]
#            theta0 = [0.00, 0.00, 0.00, 0.00, 1.57] # for TEST
#            theta1 = [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
        else:
            theta0 = [0.03, 0.06, 0.13, 0.26, 0.54]
            theta1 = [0.01, 0.00, 0.00, 0.00, 0.01, 0.01, 0.01, 0.03, 0.04, 0.13]

# angles
    phi = alpha * b * (0.5)**n
#    phi = 2. * math.asin(math.sqrt(1.00)) # for TEST


# multi control gates
    def ccry(th):
        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)
        qc.cry(-th*0.5, 0, 2)
        qc.ccx(0, 1, 2)
        qc.cry(th*0.5, 0, 2)
        return qc.to_gate(label='ccry')

    def mcry(n, th):
        qc = QuantumCircuit(n+1)
        qc.mct(list(range(n)), n)
        qc.ry(-th*0.5, n)
        qc.mct(list(range(n)), n)
        qc.ry(th*0.5, n)
        return qc.to_gate(label='mcry')

# Make operator P, R, R†
    def P(qc):
        qc.h(range(n))
        return qc
    def R(qc):
        qc.ry(phi, n)
        for q in range(n):
            qc.cry(theta0[q], q, n)
        for q in range(1, n):
            for r in range(q):
                qc.append(ccry(theta1[q*(q-1)//2+r]), [r, q, n])
        return qc
    def Rdg(qc):
        for q in reversed(range(1, n)):
            for r in reversed(range(q)):
                qc.append(ccry(-theta1[q*(q-1)//2+r]), [r, q, n])
        for q in range(n):
            qc.cry(-theta0[q], q, n)
        qc.ry(-phi, n)
        return qc
    
# Make A = R(p*I)
    def A(qc):
        P(qc)
        R(qc)
        return qc
    def Adg(qc):
        Rdg(qc)
        P(qc)
        return qc

# Sx
    def Sx(qc):
        qc.z(n)

    def cQ():
        qc = QuantumCircuit(n + 2) # n: label, n+1: control
        c = n + 1
        qc.cz(c, n) #cSx
        # cA†
        for q in reversed(range(1, n)):
            for r in reversed(range(q)):
                qc.append(mcry(3, -theta1[q*(q-1)//2+r]), [c, r, q, n])
        for q in reversed(range(n)):
            qc.append(ccry(-theta0[q]), [c, q, n])
        qc.cry(-phi, c, n)
        for q in reversed(range(n)):
            qc.ch(c, q)
        # S0 = X†CZ†X
        for q in range(n + 1):
            qc.cx(c, q)
        qc.ch(c, n)
        qc.mct(list(range(n))+[c], n)
        qc.ch(c, n)
        for q in reversed(range(n + 1)):
            qc.cx(c, q)
        # cA
        for q in range(n):
            qc.ch(c, q)
        qc.cry(phi, c, n)
        for q in range(n):
            qc.append(ccry(theta0[q]), [c, q, n])
        for q in range(1, n):
            for r in range(q):
                qc.append(mcry(3, -theta1[q*(q-1)//2+r]), [c, r, q, n])
        return qc.to_gate(label='cQ')

    def qft_dag(t):
        """t-qubit QFTdagger"""
        qc = QuantumCircuit(t)
        # Don't forget the Swaps!
        for q in range(t//2):
            qc.swap(q, t-q-1)
        for j in range(t):
            for m in range(j):
                qc.cp(-math.pi/float(2**(j-m)), m, j)
            qc.h(j)
        return qc.to_gate(label='QFT†')

# Create QuantumCircuit
    m = n + 1
    circ = QuantumCircuit(m + t, t) # Circuit with m+t qubits and t classical bits
    A(circ)
# Initialise counting qubits to |+>
    circ.h(range(m, m + t))

    reps = 1
    for count in range(m, m + t):
        cQ_q = QuantumRegister(m + 1)
        cQ_circ = QuantumCircuit(cQ_q, name='cQ^'+str(reps))
        for i in range(reps):
            cQ_circ.append(cQ(), list(range(m + 1)))
        cQ_inst = cQ_circ.to_instruction()
        circ.append(cQ_inst, list(range(m))+[count])
        reps *= 2

    circ.append(qft_dag(t), range(m, m + t))

# Measure counting qubits
    circ.measure(range(m, m + t), range(t))

# Display the circuit
    print(circ.draw())
    
# Execute and see results
    job = execute(circ, backend=q_back, shots=shots)
    result = job.result()
    counts = result.get_counts()
    print("Total counts are:",counts)

    measured_str = counts.most_frequent()
#    measured_str = max(counts, key=counts.get)
    measured_int = int(measured_str,2)
    print("Max Register Output = %i" % measured_int)
    
    theta = (measured_int/(2**t))*math.pi*2
    print("Theta = %.5f" % theta)

    N = 2**n
    M = N * (math.sin(theta*0.5)**2)
    print("No. of Solutions = %.1f" % (N-M))
    print("S(f): %.2f" % (2 * b * (1-math.sin(theta*0.5)**2)))
    if(args.fcn=='linear'):
        print('Strict value: 0.7854')
    else:
        print('Strict value: 0.5236')

    ub = t - 1 # Upper bound: Will be less than this
    err = 2*b*(math.sqrt(2*math.sin(theta*0.5)**2)\
               + 1/(2**(ub+1)))*(2**(-ub))
    print("Error < %.2f" % err)
    
    # Draw counts
    w = len(counts.keys()) * 0.6
    if(w > 30):
        w = 30
    elif(w < 4):
        w = 4
    plt.figure(figsize=(w, 8.0), dpi=72)
    plt.bar([k for k in counts.keys()], [v for v in counts.values()])
    for k, v in counts.items():
        plt.text(k, v, v, ha='center', va='bottom')
    plt.show()
