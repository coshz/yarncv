import torch 


def eval_metrics(dataloader, model):
    nc = model.num_classes()
    TP, TN, FP, FN = [0]*nc, [0]*nc, [0]*nc, [0]*nc
    model.eval()
    with torch.no_grad():
        device_ = model.device()
        for IMAGEs, LABELs in dataloader:
            IMAGEs, LABELs = IMAGEs.to(device_), LABELs.to(device_)
            PREDICTEDs = torch.argmax(model(IMAGEs), 1)
            for c in range(nc):
                TP[c] += ((PREDICTEDs == c) & (LABELs == c)).sum().item()
                TN[c] += ((PREDICTEDs != c) & (LABELs != c)).sum().item()
                FP[c] += ((PREDICTEDs == c) & (LABELs != c)).sum().item()
                FN[c] += ((PREDICTEDs != c) & (LABELs == c)).sum().item()

        accuracys = [ (TP[c] + TN[c]) / (TP[c] + TN[c] + FP[c] + FN[c]) if (TP[c] + TN[c] + FP[c] + FN[c]) > 0 else 0 for c in range(nc)]
        recalls = [ TP[c] / (TP[c] + FN[c]) if (TP[c] + FN[c]) > 0 else 0 for c in range(nc)]
        precisions = [ TP[c] / (TP[c] + FP[c]) if (TP[c] + FP[c]) > 0 else 0 for c in range(nc)]
        f1_scores = [ 2 * (precisions[c] * recalls[c]) / (precisions[c] + recalls[c]) if (precisions[c] + recalls[c]) > 0 else 0 for c in range(nc)]
        total_acc = sum(TP) / (sum(TP) + sum(FN))   
    
    d_metric = {
        'accuracy': accuracys,
        'recall': recalls,
        'precision': precisions,
        'f1_score': f1_scores,
        'total_acc': total_acc,
    }
    return d_metric


def eval_acc(dataloader, model):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        device_ = model.device()
        for IMAGEs, LABELs in dataloader:
            IMAGEs, LABELs = IMAGEs.to(device_), LABELs.to(device_)
            PREDICTEDs = torch.argmax(model(IMAGEs), 1)
            total += LABELs.size(0)
            correct += (PREDICTEDs==LABELs).sum().item()
    acc = 100.0 * correct/total
    return acc