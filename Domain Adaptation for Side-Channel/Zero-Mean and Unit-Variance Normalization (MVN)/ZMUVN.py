import numpy as np
import matplotlib.pyplot as plt
from tqdm import tnrange
import os, sys
import argparse

def load(path,length):
        # Load the npz file
    print(path)
    data = np.load(path)

    # Extract the data from the npz file
    power_trace = data['power_trace']
    plain_text = data['plain_text']
    key= data['key']
    power_trace=power_trace[:length]
    plain_text=plain_text[:length]
    np.shape(power_trace)
    np.shape(plain_text)

    return power_trace,plain_text,key

def ZMUVN(power_trace,power_trace1,n):    
    print('[LOG---- APPLYING ZMUVN.............]')
    indx=min(len(power_trace), len(power_trace1))
    trainMean = np.mean(power_trace[:140000], axis = 0)
    testMean = np.mean(power_trace1[:10000], axis = 0)
    print(np.shape(trainMean))
    print(np.shape(testMean))
    trainStd = np.std(power_trace[:140000], axis = 0)
    testStd = np.std(power_trace1[:10000], axis = 0)
    print(np.shape(trainStd))
    print(np.shape(testStd))
    coeff=(trainStd/testStd)
    modified_trace=[]
    for i in range(n):
        power_trace1[i]
        t=[]
        for j in range(5000):
            t.append((((power_trace1[i][j]-testMean[j])*coeff[j])+trainMean[j]))
        modified_trace.append(t)
    print('[LOG----ZMUVN COMPLETE.............]')

    return modified_trace




def main(opts):
    source_traces,source_plain_text, source_key=load(opts.source,opts.trace_num)
    target_traces,target_plain_text, target_key=load(opts.target,opts.test_trace_num)
    modified_trace=ZMUVN(source_traces,target_traces,opts.test_trace_num)
    print("FINAL SHAPE:",np.shape(modified_trace))
    print(target_key)
    print(source_key)
    os.makedirs(opts.output, exist_ok=True)
    modelDir = os.path.join(opts.output, 'Test.npz')
    print(modelDir)
    np.savez(modelDir,power_trace=modified_trace,plain_text=target_plain_text,key=target_key)

def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--source', help='')
    parser.add_argument('-t', '--target', help='')
    parser.add_argument('-o', '--output', help='')
    parser.add_argument('-tns', '--trace_num', type=int, default=140000, help='')
    parser.add_argument('-tnt', '--test_trace_num', type=int, default=10000, help='')

    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parseArgs(sys.argv)
    main(opts)
