import torch.nn as nn
from data_loader import get_loader
from model import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from utils import *
from prepro.build_vocab import *
from prepro.pick_image import make_dir
from infer import infer_caption_by_master, infer_caption
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

        # Get decoder's previous state
        old_embed = decoder.embed.weight.data[:4124]
        old_weight = decoder.linear.weight.data[:4124]
        old_bias = decoder.linear.bias.data[:4124]

        # Initialize new embedding and linear layers
        decoder.embed = nn.Embedding(vocab_size, args.embed_size)
        decoder.linear = nn.Linear(args.hidden_size, vocab_size)

        if args.freeze_cri or args.lwf or args.distill:
            # Assign old neurons to the newly-initialized layer, fine-tuning only should ignore this
            print("Assigning old neurons of embedding and linear layer to new decoder...")

            # Init by decoder's params
            decoder.embed.weight.data[:4124, :] = old_embed  # 4124 is the vocab size of S19
            decoder.linear.weight.data[:4124] = old_weight
            decoder.linear.bias.data[:4124] = old_bias

        encoder.to(device)
        decoder.to(device)

    else:
        # Normal training procedure
        encoder = EncoderCNN(args.embed_size).to(device)
        decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)

    if args.freeze_enc:
        args.task_name += '_freeze_enc'
    elif args.freeze_dec:
        args.task_name += '_freeze_dec'
    elif args.freeze_cri:
        args.task_name += '_freeze_cri'
    elif args.lwf:
        args.task_name += '_lwf'
    elif args.distill and args.kd1:
        args.task_name += '_kd1'
    elif args.distill and args.kd2:
        args.task_name += '_kd2'

    if args.task_type == 'seq':
        args.model_path = cfg['model']['model_path_format'].format(args.task_type, args.task_name + '_seq', 'models')
        args.cpkt_path = cfg['model']['model_path_format'].format(args.task_type, args.task_name + '_seq', 'best')
    else:
        args.model_path = cfg['model']['model_path_format'].format(args.task_type, args.task_name, 'models')
        args.cpkt_path = cfg['model']['model_path_format'].format(args.task_type, args.task_name, 'best')

    # Create model directory
    make_dir(args.model_path)

    # Pseudo-labeling option
    if args.lwf:
        print("Running pseudo-labeling option...")
        # Infer pseudo-labels using previous model
        pseudo_labels = infer_caption(img_path=train_root,
                                      json_path=train_json,
                                      model=args.check_point,
                                      vocab_path=vocab_path,
                                      prediction_path=None,
                                      id2class_path=None)

        # Freeze LSTM and decoder for later joint optimization
        for param in decoder.lstm.parameters():
            param.requires_grad_(False)
        for param in encoder.parameters():
            param.requires_grad_(False)

        data = append_json(pseudo_labels, train_json)

        # Create a new json file from the train_json
        train_json = args.caption_path + 'captions_train_lwf.json'
        with open(train_json, 'w') as file:
            json.dump(data, file)

        # Knowledge distillation option
    if args.distill:
        print("Running knowledge distillation...")
        # Teacher
        teacher_cnn = checkpoint['encoder']
        teacher_lstm = checkpoint['decoder']
        teacher_cnn.train()
        teacher_lstm.train()

        # Initialize a totally new captioning model - Student
        encoder = EncoderCNN(args.embed_size).to(device)
        decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)

        # Student
        student_cnn = encoder
        student_lstm = decoder

        # Move teacher to cuda
        teacher_cnn.to(device)
        teacher_lstm.to(device)

        # Loss between GT caption and the prediction
        criterion_lstm = nn.CrossEntropyLoss()
        # Loss between predictions of teacher and student
        criterion_distill = nn.MSELoss()

        # Params of student
        params_st = list(student_lstm.parameters()) + list(student_cnn.parameters())

        optimizer_lstm = torch.optim.Adam(params_st, lr=1e-4)
        optimizer_distill = torch.optim.Adam(student_cnn.parameters(), lr=1e-5)

    if args.freeze_enc:
        print("Freeze encoder technique!")
        for param in encoder.parameters():
            param.requires_grad_(False)

    if args.freeze_dec:
        print("Freeze decoder technique!")
        for param in decoder.lstm.parameters():
            param.requires_grad_(False)

    if args.freeze_cri:
        print("Critical Freezing technique!")
        layer_idx = -1
        for child in encoder.resnet.children():
            layer_idx += 1
            if layer_idx == 0 or layer_idx == 4:  # blk 1 & 2
                for param in child.parameters():
                    param.requires_grad = False

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

        if args.distill:
            print("Training with distillation option!")
            train_step, train_loss_step = train_distill(epoch, train_loader=train_loader,
                                                        student_cnn=student_cnn,
                                                        student_lstm=student_lstm,
                                                        teacher_cnn=teacher_cnn,
                                                        teacher_lstm=teacher_lstm,
                                                        criterion_lstm=criterion_lstm,
                                                        criterion_distill=criterion_distill,
                                                        optimizer_lstm=optimizer_lstm,
                                                        optimizer_distill=optimizer_distill)
            # Validate after an epoch
            recent_val_loss, val_step, val_loss_step = validate(epoch, val_loader=val_loader,
                                                                encoder=student_cnn,
                                                                decoder=student_lstm,
                                                                criterion=criterion)
        else:

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

    if args.lwf:
        # Make all trainable
        for param in decoder.linear.parameters():
            param.requires_grad_(True)
        for param in decoder.embed.parameters():
            param.requires_grad_(True)
        for param in decoder.lstm.parameters():
            param.requires_grad_(True)
        for param in encoder.parameters():
            param.requires_grad_(True)

        print("Unfreezing parameters ...")

        print("Critical Freezing technique!")
        layer_idx = -1
        for child in encoder.resnet.children():
            layer_idx += 1
            if layer_idx == 0 or layer_idx == 4:  # blk 1 & 2
                for param in child.parameters():
                    param.requires_grad = False

        # Joint optimization starts

        early_stopping = EarlyStopping(patience=patience, verbose=True)
        for epoch in range(args.num_epochs):
            train_step, train_loss_step = train(epoch, train_loader=train_loader,
                                                encoder=encoder,
                                                decoder=decoder,
                                                criterion=criterion,
                                                optimizer=optimizer,
                                                first_training=False,
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


def train_distill(epoch, train_loader, student_cnn, student_lstm, teacher_cnn, teacher_lstm,
                  criterion_lstm, criterion_distill, optimizer_lstm, optimizer_distill):
    """
    Train function for distillation option
    :param epoch: num of epoch for training
    :param train_loader: training loader
    :param student_cnn: student encoder
    :param student_lstm: student decoder
    :param teacher_cnn: teacher encoder
    :param teacher_lstm: teacher decoder
    :param criterion_lstm: normal loss calculation
    :param criterion_distill: loss calculation for distill part
    :param optimizer_lstm: normal optimizer
    :param optimizer_distill: optimizer for distill part
    :return:
    """

    step = []
    loss_step = []
    # Train mode on
    total_step = len(train_loader)
    student_cnn.to(device)
    student_lstm.to(device)
    student_cnn.train()
    student_lstm.train()

    for param in student_cnn.parameters():
        param.requires_grad_(True)

    for i, (images, captions, lengths) in enumerate(train_loader):
        # Set mini-batch dataset
        images = images.to(device)
        captions = captions.to(device)
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

        # Forward, backward and optimize
        optimizer_lstm.zero_grad()
        optimizer_distill.zero_grad()

        features_tr, _, _ = teacher_cnn(images)
        features_st, _, _ = student_cnn(images)

        outputs = student_lstm(features_st, captions, lengths)
        outputs_tr = teacher_lstm(features_tr, captions, lengths)

        # Add CNN distillation loss here
        lstm_loss = criterion_lstm(outputs, targets)
        if args.kd2:
            dis_loss = criterion_distill(outputs, outputs_tr)
            #print("Running KD2")
        elif args.kd1:
            dis_loss = criterion_distill(features_tr, features_st)
            #print("Running KD1")
        else:
            assert False, "Choose KD1 or KD2 option! Terminating............"
        loss = lstm_loss + dis_loss

        loss.backward()

        optimizer_lstm.step()
        optimizer_distill.step()

        # Print log info
        if i % args.log_step == 0:
            print('Training: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, LSTM Loss: {:.4f}, Distillation Loss: {:.4f}'
                  .format(epoch + 1, args.num_epochs, i, total_step, loss.item(), lstm_loss.item(), dis_loss.item()))
            step.append(i)
            loss_step.append(loss.item())

    torch.save(student_lstm.state_dict(), os.path.join(
        args.model_path, 'decoder-{}.ckpt'.format(epoch + 1)))
    torch.save(student_cnn.state_dict(), os.path.join(
        args.model_path, 'encoder-{}.ckpt'.format(epoch + 1)))

    return step, loss_step


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
        features, _, _ = encoder(images)
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
        images = images.to(device)
        captions = captions.to(device)
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

        # Forward, backward and optimize
        features, _, _ = encoder(images)
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
    parser.add_argument('--batch_size',  type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)

    parser.add_argument('--task_name',          type=str, default='2to21')
    parser.add_argument('--check_point',        type=str,
                        default='models/one/2to21/best/BEST_checkpoint_ms-coco.pth.tar')
    parser.add_argument('--check_point_vocab',  type=str,
                        default='data/vocab/2to21/vocab.pkl')

    # Technique options
    parser.add_argument('--fine_tuning', action="store_true", help="use Fine-tuning from a check point")
    parser.add_argument('--freeze_enc',  action="store_true", help="use Freezing the encoder")
    parser.add_argument('--freeze_dec',  action="store_true", help="use Freezing the decoder")
    parser.add_argument('--freeze_cri',  action="store_true", help="use Critical Freezing method")
    parser.add_argument('--lwf',         action="store_true", help="use Learning without forgetting")
    parser.add_argument('--distill',     action="store_true", help="use KD")
    parser.add_argument('--kd1',         action="store_true", help="use Knowledge distillation on intermediate space")
    parser.add_argument('--kd2',         action="store_true", help="use Knowledge distillation on output layer")

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
                if args.freeze_enc:
                    args.check_point = cfg['model']['check_point_format_seq'].format(
                        task_list[i - 1] + '_freeze_enc_seq')
                elif args.freeze_dec:
                    args.check_point = cfg['model']['check_point_format_seq'].format(
                        task_list[i - 1] + '_freeze_dec_seq')
                elif args.freeze_cri:
                    args.check_point = cfg['model']['check_point_format_seq'].format(
                        task_list[i - 1] + '_freeze_cri_seq')
                elif args.lwf:
                    args.check_point = cfg['model']['check_point_format_seq'].format(
                        task_list[i - 1] + '_lwf_seq')
                elif args.kd1:
                    args.check_point = cfg['model']['check_point_format_seq'].format(
                        task_list[i - 1] + '_kd1_seq')
                elif args.kd2:
                    args.check_point = cfg['model']['check_point_format_seq'].format(
                        task_list[i - 1] + '_kd2_seq')

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
