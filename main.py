from config import *
import os
import argparse
from torch.utils.data import Dataset, SequentialSampler, DataLoader, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup
import torch

torch.multiprocessing.set_sharing_strategy('file_system')


def main_worker(local_rank, parsed_args):
    """

    """
    if parsed_args.root_dir:
        BasicArgs.root_dir = parsed_args.root_dir

    if os.path.exists(parsed_args.cf):
        cf = import_filename(parsed_args.cf)
        Net, args, inner_collect_fn = cf.Net, cf.args, cf.inner_collect_fn

        dataset_cf = import_filename(args.dataset_cf)
        BaseDataset = dataset_cf.BaseDataset
    else:
        raise NotImplementedError('Config filename %s does not exist.' % parsed_args.cf)

    args.do_train = parsed_args.do_train
    args.n_gpu = torch.cuda.device_count()
    args.eval_save_filename = parsed_args.eval_save_filename
    if parsed_args.checkpoint_path:
        args.pretrained_model = parsed_args.checkpoint_path

    # Note that tbs/ebs is the Global batch size = GPU_PER_NODE * NODE_COUNT * LOCAL_BATCH_SIZE
    if parsed_args.tbs:
        args.train_batch_size = parsed_args.tbs
    if parsed_args.ebs:
        args.eval_batch_size = parsed_args.ebs

    if local_rank == -1:  # Do not use distributed training.
        args.rank = -1
        os.environ['RANK'] = os.environ['LOCAL_RANK'] = '-1'
        args.local_train_batch_size = args.train_batch_size
        args.local_eval_batch_size = args.eval_batch_size
    else:  # Use torch.distributed.launch for training
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.rank = int(os.environ['RANK'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.nodes = int(args.world_size / args.n_gpu)
        args.node_id = int(args.rank / args.n_gpu)
        if parsed_args.dist:
            logger.info('[node:{}/{} rank:{}/{} local_rank:{}/{}] launches'.format(
                args.node_id, args.nodes, args.rank, args.world_size, args.local_rank, args.n_gpu))
            dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=args.rank)
            args.local_train_batch_size = args.train_batch_size // args.world_size
            args.local_eval_batch_size = args.eval_batch_size // args.world_size
        else:
            raise ValueError('You must specify distributed training method.')

    # init model
    model = Net(args)
    logger.info('Successfully built model with %s parameters' % get_parameters(model))

    if parsed_args.do_train:
        logger.warning("Do training...")
        # Prepare Dataset.
        train_dataset = BaseDataset(args, split='train')
        eval_dataset = BaseDataset(args, split='val')

        if parsed_args.dist:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas=args.world_size,
                rank=args.rank,
                shuffle=True)
            eval_sampler = torch.utils.data.distributed.DistributedSampler(
                eval_dataset,
                num_replicas=args.world_size,
                rank=args.rank,
                shuffle=False)
        else:
            train_sampler = RandomSampler(train_dataset)
            eval_sampler = SequentialSampler(eval_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                      batch_size=args.local_train_batch_size,
                                      num_workers=args.num_workers)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                     batch_size=args.local_eval_batch_size,
                                     num_workers=args.num_workers)

        # Define Optimizer and Scheduler.
        optimizer = AdamW([p for n, p in model.named_parameters()], lr=args.learning_rate, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(len(train_dataloader) * args.epochs * getattr(args, 'warmup_ratio', 0.05)),
            num_training_steps=len(train_dataloader) * args.epochs
        )
        optimizers = getattr(model, 'optimizers', [optimizer])
        scheduler = getattr(model, 'scheduler', scheduler)

        trainer = Trainer(args=args, model=model, optimizers=optimizers, scheduler=scheduler,
                          pretrained_model=getattr(args, 'pretrained_model', None),
                          use_amp=getattr(args, 'use_amp', False),
                          find_unused_parameters=getattr(args, 'find_unused_parameters', False))

        trainer.train(train_loader=train_dataloader, eval_loader=eval_dataloader, epochs=args.epochs,
                      eval_step=getattr(args, 'eval_step', 5), save_step=getattr(args, 'save_step', 5),
                      resume=args.resume, use_tqdm=True,
                      max_norm=getattr(args, 'max_norm', None),
                      gradient_accumulate_steps=getattr(args, 'gradient_accumulate_steps', 1),
                      inner_collect_fn=cf.inner_collect_fn,
                      best_metric_fn=getattr(args, 'best_metric_fn', lambda x: x['train']['loss_total']))

    if parsed_args.eval_visu:
        logger.warning("Do eval_visu...")
        eval_dataset = BaseDataset(args, split='val')
        if local_rank == -1:
            eval_sampler = SequentialSampler(eval_dataset)
        else:
            eval_sampler = torch.utils.data.distributed.DistributedSampler(
                eval_dataset,
                num_replicas=args.world_size,
                rank=args.rank,
                shuffle=False)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.local_eval_batch_size,
                                     num_workers=args.num_workers)

        trainer = Trainer(args=args, model=model, pretrained_model=getattr(args, 'pretrained_model', None),
                          find_unused_parameters=getattr(args, 'find_unused_parameters', False))
        trainer.eval(eval_dataloader, inner_collect_fn=cf.inner_collect_fn, use_tqdm=True)


def add_custom_arguments(parser):
    parser.add_argument('--root_dir', default=None, type=str)
    parser.add_argument('--cf', default='./config/t2i/t2i4ccF8S256.py', type=str)
    parser.add_argument('--dist', action='store_true')
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--eval_visu', action='store_true')
    parser.add_argument('--eval_save_filename', default='eval_visu', type=str)
    parser.add_argument('--tbs', default=None, type=int)
    parser.add_argument('--ebs', default=None, type=int)
    parser.add_argument('--checkpoint_path', default=None, type=str)

    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local-rank', type=int, default=-1)
    parser = add_custom_arguments(parser)
    parsed_args = parser.parse_args()
    if parsed_args.dist:
        logger.warning('Distributed Training.')
        # main_worker(parsed_args.local_rank, parsed_args)
        main_worker(int(os.environ['LOCAL_RANK']), parsed_args)
    else:
        logger.warning('Common Training.')
        main_worker(-1, parsed_args)
