import torch
import torchattacks
from tqdm import tqdm
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def q_probs(output,label):
    first_prob = output.topk(2,dim=1).values
    #second_prob = [j[i] for i,j in enumerate(torch.index_select(output,1,label))]
    return (first_prob[:,0] / first_prob[:,1]).detach().cpu()


def q_ranking(net,loader):
    net.cuda()
    correctness = []
    all_probs = []
    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, predicted_bngn = outputs.max(1)
        predicted_bngn = predicted_bngn.eq(labels).cpu()
        correct_bngn = (predicted_bngn==True)
        correctness.append(correct_bngn)
        probs = q_probs(F.softmax(outputs),labels)
        all_probs.append(probs)
    correctness = torch.cat(correctness)
    all_probs = torch.cat(all_probs)
    net.cpu()
    return correctness, all_probs

def confidence_ranking(net,adv_loader):
    net.cuda()
    all_probs = []
    for batch_idx, (inputs, labels) in enumerate(adv_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        outputs = F.softmax(outputs,1)

        correct_confidence = outputs.gather(1,labels.unsqueeze(1)).squeeze().detach().cpu()
        all_probs.append(correct_confidence)
    all_probs = torch.cat(all_probs).cpu()
    net.cpu()
    return 0, all_probs


def extract_probs(output,label):
    predicted_prob = output.max(1).values
    true_prob = [j[i] for i,j in enumerate(torch.index_select(output,1,label))]
    return predicted_prob - torch.tensor(true_prob)

def test_transferability_to_sur_conf_diff(net,loader):
    atk = torchattacks.PGD(net,pert_size,steps=20)
    correctness = []
    all_probs = []
    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)
        adv_inputs = atk(inputs, labels)
        outputs = net(adv_inputs)
        _, predicted_bngn = outputs.max(1)
        predicted_bngn = predicted_bngn.eq(labels).cpu()
        correct_bngn = (predicted_bngn==True)
        correctness.append(correct_bngn)
        outputs_sur = net(adv_inputs).detach().cpu() 
        probs = extract_probs(F.softmax(outputs_sur),labels.detach().cpu())
        all_probs.append(probs)
    correctness = torch.cat(correctness)
    all_probs = torch.cat(all_probs)
    return correctness, all_probs


def min_pert_ranking(net,loader):
    atks = []
    for i in range(0,12):
        atks.append(torchattacks.PGD(net,i/255,steps=7))
    min_perts = []
    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)
        min_pert= torch.ones(inputs.shape[0])*100
        for i, atk in enumerate(atks):
            adv_inputs = atk(inputs, labels)
            pred = net(adv_inputs).max(1)[1]
            pred_correct = pred.eq(labels).cpu()
            min_pert[~pred_correct] = torch.minimum(min_pert[~pred_correct],torch.ones(min_pert[~pred_correct].shape)*i)
        min_perts+=list(min_pert)
    return 0,min_perts


def noise_pert_ranking(net,loader,pert_size):
    net.cuda()
    ranking = []
    for batch_idx, (inputs, labels) in enumerate(tqdm(loader)):
        pred_corrects = []
        inputs, labels = inputs.to(device), labels.to(device)

        for i in range(20):
            noise = torch.normal(0.0,1.0,shape=inputs.shape).cuda()*(pert_size) 
            inputs_noised= inputs+noise
            pred = net(inputs_noised).max(1)[1]
            pred_correct = pred.eq(labels).cpu()
            pred_corrects.append(pred_correct)
        ranking += (torch.stack(pred_corrects).float().mean(0)).cpu()
    net.cpu()
    return 0,ranking

def pen_variance_rank(net,loader):
    ranking = []
    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device),  labels.to(device)
        _, pens = get_best_init(net,inputs,True)
        var_pens = get_mean_of_var_pens(pens)
        ranking.append(var_pens)
    ranking = torch.cat(ranking).detach().cpu().numpy()
    return 0, ranking
def test_transferability_to_var(net,loader):
    atk = torchattacks.PGD(net,pert_size,steps=20)
    correctness = []
    variances = []
    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device),  labels.to(device)
        adv_inputs = atk(inputs, labels)
        outputs = net(adv_inputs)
        _, predicted_bngn = outputs.max(1)
        predicted_bngn = predicted_bngn.eq(labels).cpu()
        correct_bngn = (predicted_bngn==True)
        _, pens = get_best_init(net,inputs,True)
        var_pens = get_mean_of_var_pens(pens)
        correctness.append(correct_bngn)
        variances.append(var_pens)
    correctness = torch.cat(correctness)
    variances =  torch.cat(variances)
    return correctness, variances

def get_pert_size(adv_data,testloader):
    pert_sizes = []
    for (adv,_),(clean,_) in zip (adv_data,testloader):
        pert_sizes.append((adv - clean).abs().mean([1,2,3]))
    return torch.cat(pert_sizes)
