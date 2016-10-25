import pylab
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("result")
args = parser.parse_args()

with open(args.result, 'r') as f:
    data = json.load(f)
    #print json.dumps(data, sort_keys = True, indent = 4)
    train_iterations = [d["iteration"] for d in data]
    train_accuracies = [d["main/accuracy"] for d in data]
    train_losses = [d["main/loss"] for d in data]
    val_iterations = [d["iteration"] for d in data if "validation/main/accuracy" in d]
    val_accuracies = [d["validation/main/accuracy"] for d in data if "validation/main/accuracy" in d]
    val_losses = [d["validation/main/loss"] for d in data if "validation/main/loss" in d]

    pylab.xlabel("iteration")
    pylab.ylabel("accuracy")
    pylab.ylim(ymin=0)
    pylab.ylim(ymax=1)
    pylab.plot(train_iterations, train_accuracies, label="train")
    pylab.plot(val_iterations, val_accuracies, label="validation")
    pylab.legend(loc='lower right')
    pylab.show()
