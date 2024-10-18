import argparse

parser = argparse.ArgumentParser(description='VadCLIP')
parser.add_argument('--seed', default=234, type=int)

parser.add_argument('--embed-dim', default=512, type=int)
parser.add_argument('--visual-length', default=256, type=int)
parser.add_argument('--visual-width', default=512, type=int)
parser.add_argument('--visual-head', default=1, type=int)
parser.add_argument('--visual-layers', default=2, type=int)
parser.add_argument('--attn-window', default=8, type=int)
parser.add_argument('--prompt-prefix', default=10, type=int)
parser.add_argument('--prompt-postfix', default=10, type=int)
parser.add_argument('--classes-num', default=14, type=int)

parser.add_argument('--max-epoch', default=10, type=int)
parser.add_argument('--model-path', default='../model/model_ucf.pth')
parser.add_argument('--use-checkpoint', default=False, type=bool)
parser.add_argument('--checkpoint-path', default='../model/checkpoint.pth')
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--train-list', default='../list/ucf_CLIP_rgb.csv')
parser.add_argument('--test-list', default='../list/ucf_CLIP_rgbtest.csv')
parser.add_argument('--gt-path', default='../list/gt_ucf.npy')
parser.add_argument('--gt-segment-path', default='../list/gt_segment_ucf.npy')
parser.add_argument('--gt-label-path', default='../list/gt_label_ucf.npy')

parser.add_argument('--lr', default=2e-5)
parser.add_argument('--scheduler-rate', default=0.1)
parser.add_argument('--scheduler-milestones', default=[4, 8])

##################################################################################################################
# 超参数配置
parser.add_argument('--k_easy', default=30, type=int, help="Number of easy snippets")
parser.add_argument('--k_hard', default=5, type=int, help="Number of hard snippets")
parser.add_argument('--M', default=3, type=int, help="Erosion/Dilation parameter M")
parser.add_argument('--m', default=24, type=int, help="Erosion/Dilation parameter m")
parser.add_argument('--loss_type', default="neg_log", help="")
parser.add_argument('--metric', default="Euclidean", help="")
# 对比损失相关超参数
parser.add_argument('--alpha1', default=0.3, type=float, help="Weight for action contrastive loss")
parser.add_argument('--alpha2', default=0.4, type=float, help="Weight for background contrastive loss")

parser.add_argument('--alpha5', default=0.1, type=float, help="Weight for action contrastive loss")
parser.add_argument('--alpha6', default=0.1, type=float, help="Weight for background contrastive loss")

# 测试