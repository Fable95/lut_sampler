import subprocess
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num_parties',  action='store', dest='n', type = int, required = False)
parser.add_argument('-ps','--pool_size',  action='store', dest='ps', type = int, required = False)
parser.add_argument('-l', '--lambda',  action='store', dest='l', type = int, required = False)
args = parser.parse_args()

def test(n, ps):
    commands = []
    for i in range(n):
        s = "./build/bin/test_vec_gen " + ( i + 1).__str__() + " 80000 " + n.__str__()
        if ps:
            s += " " + ps.__str__()
        if args.l:
            s+= " " + args.l.__str__()
        if i != n-1: 
            s += " & "
        commands.append(s)
    print(commands)
    return commands


parties = [4]
start = 6
stop = 7

for parties in parties:
    print(f"Test {parties} parties")
    for pool_size in range(start,stop):   
        print(f"Start with pool size {pool_size}")
        procs = []
        commands = test(parties, pool_size)
        for c in commands:
            procs.append(subprocess.Popen(c, shell=True))
            time.sleep(1)
        print("waiting for processes to finish")
        for p in procs:
            p.wait()
        print("All processes done, sleeping for 5 sec")
        time.sleep(5)
