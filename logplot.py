import pylab
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("result")
parser.add_argument("--Y", '-y', choices = ("accuracy", "loss"), default="accuracy")
args = parser.parse_args()

with open(args.result, 'r') as f:
    data = json.load(f)
    #print json.dumps(data, sort_keys = True, indent = 4)
    train_iterations = [d["iteration"] for d in data]
    train_accuracies = [d["main/accuracy"] for d in data]
    train_losses = [d["main/loss"] for d in data]
    train_Y = [d["main/" + args.Y] for d in data]
    val_iterations = [d["iteration"] for d in data if "validation/main/accuracy" in d]
    val_accuracies = [d["validation/main/accuracy"] for d in data if "validation/main/accuracy" in d]
    val_losses = [d["validation/main/loss"] for d in data if "validation/main/loss" in d]
    val_Y = [d["validation/main/" + args.Y] for d in data if "validation/main/" + args.Y in d]

    pylab.xlabel("iteration")
    #pylab.ylabel("accuracy")
    pylab.ylabel(args.Y)
    if args.Y == "accuracy":
        pylab.ylim(ymin=0)
        pylab.ylim(ymax=1)
    pylab.plot(train_iterations, train_Y, label="train")
    pylab.plot(val_iterations, val_Y, label="validation", marker="o")
    if args.Y == "accuracy":
        pylab.legend(loc='lower right')
    elif args.Y == "loss":
        pylab.legend(loc='upper right')
    pylab.show()
