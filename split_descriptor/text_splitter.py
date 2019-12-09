import argparse, os, random, sys

parser = argparse.ArgumentParser(description = "Shuffle a file's lines and output it in chunks.")

parser.add_argument("infile", type=str, help="File to split.")
parser.add_argument("chunkCount", type=int, help="Number of chunks to split the lines into.")
parser.add_argument("out_dir", type=str, help="Directory to output the chunks in.")

args = parser.parse_args()

fileList = []
with open(args.infile, "r") as inpFile:
    fileList = [ln.strip() for ln in inpFile.readlines()]

outputPrefix = args.infile.split('/')[-1].split('.')[0]

if fileList == []:
    print("Failed getting input")
    sys.exit(0)

random.shuffle(fileList)

outFileLists = [[] for _ in range(args.chunkCount)]

for i, filename in enumerate(fileList):
    outFileLists[i % args.chunkCount].append(filename)

for i, filenames in enumerate(outFileLists):
    currentOutput = f"{outputPrefix}_{i}.txt"
    with open(os.path.join(args.out_dir, currentOutput), "w") as coutFile:
        for fname in filenames:
            coutFile.write(fname + "\n")

