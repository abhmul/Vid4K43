from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from kaggleutils import safe_create_file

from data import CrappifyDataset, Crappify
from layers import get_resnet
from models import GeneratorPretrain, Input
from runners import Trainer, Checkpointer
from experimenting import ExperimentParser, read_experiment

# comes with args.experiment
parser = ExperimentParser()
parser.add_argument("-d", "--data")
parser.add_argument("--crop_size", default=256, type=int)
parser.add_argument("--downsample", default=0.5, type=float)
parser.add_argument("--val_split", default=0.05, type=float)
parser.add_argument("--batch_size", default=2, type=int)

parser.add_argument("--encoder", default="resnet101")
parser.add_argument("--channels_factor", default=2, type=int)
parser.add_argument("--batchnorm", action="store_true")
parser.add_argument("--no_spectral_norm", action="store_true")
parser.add_argument("--input_batchnorm", action="store_true")

parser.add_argument("--lr", default=1e-2, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--epochs", default=50, type=int)
parser.add_argument("--train_steps", type=int, default=None)
parser.add_argument("--val_steps", type=int, default=None)

args = parser.parse_args()
args = read_experiment(args)

writer = SummaryWriter(log_dir=f"runs/{args.experiment}")

# TODO: Add noise
crappifier = Crappify(args.downsample)
data = CrappifyDataset(args.data, crappifier=crappifier, crop_size=args.crop_size)
train_data, val_data = data.split(args.val_split)

train_dataloader = DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True, num_workers=12
)
val_dataloader = DataLoader(val_data, batch_size=args.batch_size, num_workers=12)

model = GeneratorPretrain(
    get_resnet(args.encoder, pretrained=True),
    channels_factor=args.channels_factor,
    batchnorm=args.batchnorm,
    spectral_norm=(not args.no_spectral_norm),
    input_batchnorm=args.input_batchnorm,
)
model(Input(3, args.crop_size, args.crop_size))
print(model)
model.save(safe_create_file(f"../weights/{args.experiment}.state"))

checkpointer = Checkpointer(
    safe_create_file(f"../weights/{args.experiment}.state"), save_best=True
)

# Warm up the decoder first
print("Warming up decoder")
model.add_optimizer(SGD(model.decoder_parameters(), lr=1e-3, momentum=0.9), name="sgd")
trainer = Trainer(
    epochs=1, steps_per_epoch=args.train_steps, validation_steps=args.val_steps
)
for epoch, epoch_outputs in trainer(
    model,
    train_dataloader,
    validation_dataloader=val_dataloader,
    writer=writer,
    train_tag="decoder/train",
    validation_tag="decoder/validation",
):
    checkpointer(model, epoch_outputs)

# Train the full model
print("Pretraining generator")
optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
model.add_optimizer(optimizer, name="sgd")
model.add_scheduler(
    CosineAnnealingWarmRestarts(optimizer, T_0=2 * len(train_dataloader), T_mult=2),
    name="annealing",
)
trainer = Trainer(
    args.epochs, steps_per_epoch=args.train_steps, validation_steps=args.val_steps
)
for epoch, epoch_outputs in trainer(
    model, train_dataloader, validation_dataloader=val_dataloader, writer=writer
):
    checkpointer(model, epoch_outputs)


writer.close()
