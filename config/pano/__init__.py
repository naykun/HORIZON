from config import *
from config.modules import *

def inner_collect_fn(args, inputs, outputs, log_dir, epoch, eval_save_filename='eval_visu'):
    rank = int(os.getenv('RANK', '-1'))
    if rank == -1:
        splice = ''
    else:
        splice = '__' + str(rank)
    if epoch == -1:
        eval_log_dir = os.path.join(log_dir, eval_save_filename)
    else:
        eval_log_dir = os.path.join(log_dir, 'train_%05d' % (epoch + 1))

    if rank in [-1, 0]:
        logger.warning(eval_log_dir)
        ensure_dirname(eval_log_dir)

        # Save Model Setting
        type_output = [int, float, str, bool, tuple, dict, type(None), ]
        setting_output = {item: getattr(args, item) for item in dir(args) if
                          type(getattr(args, item)) in type_output and not item.startswith('__')}
        data2file(setting_output, os.path.join(eval_log_dir, 'Model_Setting.json'))

    gt_save_path = os.path.join(eval_log_dir, 'gt')
    pred_save_path = os.path.join(eval_log_dir, 'pred')

    dl = {**inputs, **{k: v for k, v in outputs.items() if k.split('_')[0] == 'logits'}}
    ld = dl2ld(dl)

    k, l = ld[0]['logits_imgs'].shape[:2]
    for idx, sample in enumerate(ld):
        prefix = sample.get('input_text', 'horizon')[:200].replace(os.sep, '') + splice
        img_name = sample.get('img_name', 'horizon')
        if isinstance(img_name, list):
            img_name = "__".join(img_name)
        prefix = prefix + "__" + img_name + splice
        postfix = '_' + str(round(time.time() * 10))

        data2file(sample['label_imgs'], os.path.join(gt_save_path, prefix + postfix + '.png'), nrow=l, normalize=True,
                  value_range=(-1, 1),
                  override=True)

        for i in range(k):
            score = '_score' + str(sample['logits_clip'][i].cpu().item())[:6] + '_' if 'logits_clip' in sample else '_'
            data2file(sample['logits_imgs'][i], os.path.join(pred_save_path, prefix + postfix + score + str(i) + '.png'),
                      nrow=l,
                      normalize=True,
                      value_range=(-1, 1),
                      override=True)

