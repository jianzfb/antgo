from antgo.framework.helper.models.nas.tutorial import AccuracyPredictor, FLOPsTable, LatencyTable, EvolutionFinder
from antgo.framework.helper.models.nas.tutorial import evaluate_ofa_subnet, evaluate_ofa_specialized
from antgo.framework.helper.models.nas.model_zoo import ofa_net
import time
from torchvision import transforms, datasets
import math
import torch
import os

def build_val_transform(size):
    return transforms.Compose([
        transforms.Resize(int(math.ceil(size / 0.875))),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

def run():
    imagenet_data_path = '/mnt/bd/jiantrainer/dataset/imagenet'
    data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            root=os.path.join(imagenet_data_path, 'val'),
            transform=build_val_transform(224)
        ),
        batch_size=250,  # test batch size
        shuffle=True,
        num_workers=16,  # number of workers for the data loader
        pin_memory=True,
        drop_last=False,
    )
    print('The ImageNet dataloader is ready.')

    ofa_network = ofa_net('ofa_mbv3_d234_e346_k357_w1.2', pretrained=True)
    print('The OFA Network is ready.')    


    target_hardware = 'note10'
    latency_table = LatencyTable(device=target_hardware)
    print('The Latency lookup table on %s is ready!' % target_hardware)

    # accuracy predictor
    accuracy_predictor = AccuracyPredictor(
        pretrained=True,
        device='cuda:0'
    )

    print('The accuracy predictor is ready!')
    print(accuracy_predictor.model)
        
    latency_constraint = 25  # ms, suggested range [15, 33] ms
    P = 100  # The size of population in each generation
    N = 500  # How many generations of population to be searched
    r = 0.25  # The ratio of networks that are used as parents for next generation
    params = {
        'constraint_type': target_hardware, # Let's do FLOPs-constrained search
        'efficiency_constraint': latency_constraint,
        'mutate_prob': 0.1, # The probability of mutation in evolutionary search
        'mutation_ratio': 0.5, # The ratio of networks that are generated through mutation in generation n >= 2.
        'efficiency_predictor': latency_table, # To use a predefined efficiency predictor.
        'accuracy_predictor': accuracy_predictor, # To use a predefined accuracy_predictor predictor.
        'population_size': P,
        'max_time_budget': N,
        'parent_ratio': r,
    }

    # build the evolution finder
    finder = EvolutionFinder(**params)

    # start searching
    result_lis = []
    st = time.time()
    best_valids, best_info, others_info = finder.run_evolution_search()
    result_lis.append(best_info)
    result_lis.extend(others_info)
    ed = time.time()
    print('Found best architecture on %s with latency <= %.2f ms in %.2f seconds! '
        'It achieves %.2f%s predicted accuracy with %.2f ms latency on %s.' %
        (target_hardware, latency_constraint, ed-st, best_info[0] * 100, '%', best_info[-1], target_hardware))

    # visualize the architecture of the searched sub-net
    _, net_config, latency = best_info
    ofa_network.set_active_subnet(ks=net_config['ks'], d=net_config['d'], e=net_config['e'])
    print('Architecture of the searched sub-net:')
    print(ofa_network.module_str)

    top1s = []
    latency_list = []
    for result in result_lis:
        acc, net_config, latency = result
        print('Evaluating the sub-network with latency = %.1f ms acc = %.6f on %s' % (latency, acc, target_hardware))
        top1 = evaluate_ofa_subnet(
            ofa_network,
            imagenet_data_path,
            net_config,
            data_loader,
            batch_size=250,
            device='cuda:0')
        print(f'top 1 {top1}')
        top1s.append(top1)
        latency_list.append(latency)