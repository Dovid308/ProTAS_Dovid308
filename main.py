import torch
from batch_gen import BatchGenerator
import os
import argparse
import random
import logging
from datetime import datetime
from model import MultiStageModel, ASFormerPROTAS, Pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

# Argument parser for command-line options
parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train', help="Action to perform: train, predict, predict_online")
parser.add_argument('--exp_id', default='mstcn', type=str, help="Experiment ID for model saving")
parser.add_argument('--backbone', default='mstcn', help="Backbone architecture to use")
parser.add_argument('--causal', action='store_true', help="Use causal architecture (no future information)")



parser.add_argument('--dataset', default="ptg", help="Dataset to use")
parser.add_argument('--split', default='1', help="Data split to use")


parser.add_argument('--batch_size', default=1, type=int, help="Batch size for training")
parser.add_argument('--num_epochs', default=50, type=int, help="Number of training epochs")
parser.add_argument('--lr', default=0.0005, type=float, help="Learning rate")


# Loss weights for auxiliary tasks, per toglierlo a zero 
parser.add_argument('--progress_lw', default=1.0, type=float, help="Loss weight for progress prediction")

# Graph-related arguments
parser.add_argument('--learnable_graph', action='store_true', help="Use learnable graph structures")
parser.add_argument('--graph', action='store_true', help="Use graph structures")
parser.add_argument('--graph_lw', default=0.1, type=float, help="Loss weight for graph prediction")


args = parser.parse_args()


bz = args.batch_size
lr = args.lr 
num_epochs = args.num_epochs


##### DATASET RELATED PARAMETER

#qui da modificare nel caso di altri dataset
is_unified = False 
if (args.dataset == "EgoPER_action_segmentation"):
   is_unified=True
# use the full temporal resolution @ 15fps -> actually we dont use salads

if (is_unified):
  mapping_file = f"./data/{args.dataset}/mapping_global.txt"
else:
    mapping_file = f"./data/{args.dataset}/mapping.txt"

sample_rate = 1
# sample input features @ 15fps instead of 30 fps
# for 50salads, and up-sample the output to 30 fps
if args.dataset == "50salads":
    sample_rate = 2

#QUI ANCORA DEVO CAPIRE BENE COSA VUOL DIRE STA ROBA, IO ANCORA NON HO TOCCATO
# Read action mapping file
with open(mapping_file, 'r') as file_ptr:
    actions = file_ptr.read().split('\n')[:-1]
# Create action dictionary
actions_dict = dict()
bg_class = 'BG' if args.dataset in ['EgoPER_action_segmentation/coffee', 'EgoPER_action_segmentation/tea', 'EgoPER_action_segmentation/pinwheels', 'EgoPER_action_segmentation/oatmeal', 'EgoPER_action_segmentation/quesadilla', 'EgoPER_action_segmentation'] else 'background'
map_delimiter = '|' if args.dataset in ['EgoPER_action_segmentation/coffee', 'EgoPER_action_segmentation/tea', 'EgoPER_action_segmentation/pinwheels', 'EgoPER_action_segmentation/oatmeal', 'EgoPER_action_segmentation/quesadilla', 'EgoPER_action_segmentation'] else ' '
feature_transpose = True if args.dataset in ['EgoPER_action_segmentation/coffee', 'EgoPER_action_segmentation/tea', 'EgoPER_action_segmentation/pinwheels', 'EgoPER_action_segmentation/oatmeal', 'EgoPER_action_segmentation/quesadilla', 'EgoPER_action_segmentation'] else False
for a in actions:
    actions_dict[a.split(map_delimiter)[1]] = int(a.split(map_delimiter)[0])
num_classes = len(actions_dict)


# Small path resolution

train_list = f"train.split{args.split}"
test_list = f"test.split{args.split}"



graph_path = f"./data/{args.dataset}/graph/graph.pkl"

##logging results
expdir = f"./results/{args.exp_id}/{args.dataset}/split_{args.split}"

# Create directories if they don't exist
os.makedirs(expdir, exist_ok=True)

# Logger setup
logger = logging.getLogger('ProTAS')
logger.setLevel(logging.DEBUG)

log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

fh = logging.FileHandler(os.path.join(expdir, log_filename), mode='w')
fh.setFormatter(formatter)
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

logger.info(args)





# Model parameters per backbone
MSTCN_PARAMS = {
    'num_stages': 4,
    'num_layers': 10,
    'num_f_maps': 64,
    'features_dim': 2048,
}

ASFORMER_PARAMS = {
    'num_decoders': 3,
    'num_layers': 9,
    'num_f_maps': 64,
    'features_dim': 2048,
    'r1': 2,
    'r2': 2,
    'channel_masking_rate': 0.3,
}

# Build model
if args.backbone == 'mstcn':
    p = MSTCN_PARAMS
    model = MultiStageModel(
        p['num_stages'], p['num_layers'], p['num_f_maps'], p['features_dim'], num_classes,
        causal=args.causal,
        use_graph=args.graph, init_graph_path=graph_path, learnable=args.learnable_graph
    )
elif args.backbone == 'asformer':
    p = ASFORMER_PARAMS
    model = ASFormerPROTAS(
        num_decoders=p['num_decoders'], num_layers=p['num_layers'], r1=p['r1'], r2=p['r2'],
        num_f_maps=p['num_f_maps'], input_dim=p['features_dim'], num_classes=num_classes,
        channel_masking_rate=p['channel_masking_rate'], causal=args.causal,
        use_graph=args.graph, init_graph_path=graph_path, learnable=args.learnable_graph
    )
else:
    raise ValueError(f"backbone non supportato: {args.backbone!r}, scegli 'mstcn' o 'asformer'")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.info(f"Backbone: {args.backbone} | Parametri totali: {total_params:,} | Trainabili: {trainable_params:,}")

# Initialize runner
pipe = Pipeline(
    model=model,
    num_classes=num_classes,
    logger=logger,
    save_dir=expdir,
    actions_dict=actions_dict,
    progress_lw=args.progress_lw,
    graph_lw=args.graph_lw,
    bg_class = bg_class,
    map_delimiter = map_delimiter
)




if args.action == "train":
    batch_gen = BatchGenerator(f"data/{args.dataset}", num_classes, actions_dict, sample_rate, feature_transpose, is_unified)
    batch_gen.read_data(train_list)
    pipe.train(batch_gen, num_epochs=num_epochs, batch_size=bz, learning_rate=lr, device=device)
    
    pipe.predict(test_list, device, sample_rate, feature_transpose, f"data/{args.dataset}", is_unified=is_unified)
    pipe.evaluate(test_list,f"data/{args.dataset}", is_unified=is_unified)


elif args.action == 'predict':
    pipe.predict(test_list, device, sample_rate, feature_transpose,f"data/{args.dataset}", is_unified=is_unified)
    pipe.evaluate(test_list,f"data/{args.dataset}", is_unified=is_unified)


elif args.action == "predict_online":
    pipe.predict_online(test_list, device, sample_rate, feature_transpose, dataset_root=f"data/{args.dataset}", is_unified=is_unified)
    pipe.evaluate(test_list, f"data/{args.dataset}", is_unified=is_unified)