import torch.nn as nn
from data_loader import get_loader
from model import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from utils import *
from prepro.build_vocab import *
from prepro.pick_image import make_dir
import numpy as np
import pickle
import argparse
import json


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg = load_and_print_cfg('config.yaml')


def main(args):
    print(args)
    epochs_since_improvement = 0

    # Create model directory
    make_dir(args.model_path)

    # Image pre-processing, normalization for the pre-trained res-net
    transform = transforms.Compose([
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper
    vocab_path = args.vocab_path
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build data loader
    train_root = args.image_dir + cfg['train']['TRAIN_DIR']
    train_json = args.caption_path + cfg['train']['train_annotation']

    val_root = args.image_dir + cfg['train']['VAL_DIR']
    val_json = args.caption_path + cfg['train']['valid_annotation']

    # After patience epochs without improvement, break training
    patience = cfg['train']['patience']
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    if args.check_point and os.path.isfile(args.check_point):
        checkpoint = torch.load(args.check_point)

    old_vocab_size = 0
    if args.fine_tuning:
        encoder = checkpoint['encoder']
        decoder = checkpoint['decoder']
        print("Fine tuning with check point is {}".format(args.check_point))

        vocab, old_vocab_size = append_vocab(args.check_point_vocab, vocab)

        with open(vocab_path, 'wb') as v:
            print("Dump {} entries to vocab {}".format(vocab.idx, vocab_path))
            pickle.dump(vocab, v)
        vocab_size = len(vocab)

        # Initialize new embedding and linear layers
        decoder.embed = nn.Embedding(vocab_size, args.embed_size)
        decoder.linear = nn.Linear(args.hidden_size, vocab_size)

        encoder.to(device)
        decoder.to(device)

    else:
        # Normal training procedure
        encoder = EncoderCNN(args.embed_size).to(device)
        decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)

    if args.task_type == 'seq':
        args.model_path = cfg['model']['model_path_format'].format(args.task_type, args.task_name + '_seq', 'models')
        args.cpkt_path = cfg['model']['model_path_format'].format(args.task_type, args.task_name + '_seq', 'best')
    else:
        args.model_path = cfg['model']['model_path_format'].format(args.task_type, args.task_name, 'models')
        args.cpkt_path = cfg['model']['model_path_format'].format(args.task_type, args.task_name, 'best')

    # Create model directory
    make_dir(args.model_path)

    train_loader = get_loader(root=train_root, json=train_json, vocab=vocab,
                              transform=transform, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)

    val_loader = get_loader(root=val_root, json=val_json, vocab=vocab,
                            transform=transform, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # Theses vars are for plotting
    avg_train_losses = []
    avg_val_losses = []

    for epoch in range(args.num_epochs):

        train_step, train_loss_step = train(epoch, train_loader=train_loader,
                                            encoder=encoder,
                                            decoder=decoder,
                                            criterion=criterion,
                                            optimizer=optimizer,
                                            first_training=True,
                                            old_vocab_size=old_vocab_size)
        # Validate after an epoch
        recent_val_loss, val_step, val_loss_step = validate(epoch, val_loader=val_loader,
                                                            encoder=encoder,
                                                            decoder=decoder,
                                                            criterion=criterion)
        train_loss = np.average(train_loss_step)
        val_loss = np.average(val_loss_step)

        avg_train_losses.append(train_loss)
        avg_val_losses.append(val_loss)

        # Save checkpoint
        make_dir(args.cpkt_path)
        early_stopping(args.cpkt_path, cfg['train']['data_name'], epoch, epochs_since_improvement, encoder, decoder, optimizer,
                       optimizer, val_loss)

        if early_stopping.early_stop:
            print("Early Stopping!")
            break

    # Uncomment this to plot loss curve
    #  loss_visualize(train_step, train_loss_step, val_step, val_loss_step)


def train(epoch, train_loader, encoder, decoder, criterion, optimizer, first_training, old_vocab_size):
    """
    Train function
    :param epoch: epoch
    :param train_loader: training loader
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss calculation
    :param optimizer: optimizer
    :param first_training: this flag is used for pseudo-labeling, we train 2 times
    :param old_vocab_size: size of the old vocab
    :return:
    """

    step = []
    loss_step = []

    # Train mode on
    total_step = len(train_loader)
    encoder.train()
    decoder.train()
    print(first_training)

    for i, (images, captions, lengths) in enumerate(train_loader):
        # Set mini-batch dataset
        images = images.to(device)
        captions = captions.to(device)
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

        # Forward, backward and optimize
        features = encoder(images)
        outputs = decoder(features, captions, lengths)
        loss = criterion(outputs, targets)
        decoder.zero_grad()
        encoder.zero_grad()
        loss.backward()

        optimizer.step()

        # Print log info
        if i % args.log_step == 0:
            print('Training: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, args.num_epochs, i, total_step, loss.item()))
            step.append(i)
            loss_step.append(loss.item())

    return step, loss_step


def validate(epoch, val_loader, encoder, decoder, criterion):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param epoch
    :return:
    """

    step = []
    loss_step = []
    loss_over_validation = 0

    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    total_step = len(val_loader)
    for i, (images, captions, lengths) in enumerate(val_loader):
        # Set mini-batch dataset
        images = images.to(device)
        captions = captions.to(device)
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

        # Forward, backward and optimize
        features = encoder(images)
        outputs = decoder(features, captions, lengths)
        loss = criterion(outputs, targets)
        loss_over_validation += loss.item()

        # Print log info
        if i % args.log_step == 0:
            print('Validation: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, args.num_epochs, i, total_step, loss.item()))
            step.append(i)
            loss_step.append(loss.item())

    return loss_over_validation, step, loss_step


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # task type is one | once | seq
    parser.add_argument('--task_type',    type=str, default='one', help='Add classes one by one or once')

    parser.add_argument('--log_step',  type=int, default=10, help='step size for printing log info')
    parser.add_argument('--save_step', type=int, default=400, help='step size for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')

    # Model parameters
    parser.add_argument('--embed_size',  type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers',  type=int, default=1, help='number of layers in lstm')

    parser.add_argument('--num_epochs',  type=int, default=50)
    parser.add_argument('--batch_size',  type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)

    parser.add_argument('--task_name',          type=str, default='2to21')
    parser.add_argument('--check_point',        type=str,
                        default='models/one/2to21/best/BEST_checkpoint_ms-coco.pth.tar')
    parser.add_argument('--check_point_vocab',  type=str,
                        default='data/vocab/2to21/vocab.pkl')

    # Technique options
    parser.add_argument('--fine_tuning', action="store_true", help="use Fine-tuning from a check point")

    # As Karpathy, 3e-4 is the best learning rate for Adam
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    args = parser.parse_args()

    args.vocab_path = cfg['dataset']['vocab_format'].format(args.task_name)
    args.image_dir = cfg['dataset']['image_dir_format'].format(args.task_name)
    args.caption_path = cfg['dataset']['caption_path_format'].format(args.task_name)
    args.model_path = cfg['model']['model_path_format'].format(args.task_type, args.task_name, 'models')
    args.cpkt_path = cfg['model']['model_path_format'].format(args.task_type, args.task_name, 'best')

    if args.task_type == 'seq':
        print("Running sequentially!")
        task_list = cfg['train']['seq_task_list']
        for i, task_name in enumerate(task_list):
            # First task (i=0) will get checkpoint from 2to21
            if i >= 1:
                args.check_point_vocab = cfg['dataset']['vocab_format'].format(task_list[i-1])
            args.task_name = task_name
            args.vocab_path = cfg['dataset']['vocab_format'].format(args.task_name)
            args.image_dir = cfg['dataset']['image_dir_format'].format(args.task_name)
            args.caption_path = cfg['dataset']['caption_path_format'].format(args.task_name)
            args.model_path = cfg['model']['model_path_format'].format(args.task_type, args.task_name + '_seq', 'models')
            args.cpkt_path = cfg['model']['model_path_format'].format(args.task_type, args.task_name + '_seq', 'best')
            main(args)
    else:
        main(args)
