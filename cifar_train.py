import os
import time
import argparse
import random
from CIFAR_10_100.data_utils import *
from CIFAR_10_100.resnet import *
from CIFAR_10_100.loss import *
import datetime

start_time = time.time()
parser = argparse.ArgumentParser(description='Imbalanced Example')
parser.add_argument('--dataset', default='cifar100', type=str,
                    help='dataset (cifar10 or cifar100[default])')
parser.add_argument('--num_classes', type=int, default=100)
parser.add_argument('--num_meta', type=int, default=100,
                    help='The number of meta data for each class.(cifar-10:10, cifar-100:100)')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--imb_factor', type=float, default=0.01)  # 0.1=100 0.02=50 0.01=100 0.005=200
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--split', type=int, default=1000)
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--lam', default=0.5, type=float, help='[0.25, 0.5, 0.75, 1.0] default=0.5')
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--meta_lr', default=0.1, type=float)
parser.add_argument('--save_name', default='name', type=str)
parser.add_argument('--idx', default='0', type=str)

args = parser.parse_args()

for arg in vars(args):
    print("{}={}".format(arg, getattr(args, arg)))
if args.seed is not None:
    print('args.seed:{}'.format(args.seed))
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True
    torch.backends.cudnn.deterministic =True
    torch.backends.cudnn.benchmark =False

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
kwargs = {'num_workers': 0, 'pin_memory': False}
use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
train_data_meta, train_data, test_dataset = build_dataset(args.dataset, args.num_meta)

print(f'length of meta dataset:{len(train_data_meta)}')
print(f'length of train dataset: {len(train_data)}')

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True, **kwargs)

classe_labels = range(args.num_classes)

data_list = {}

for j in range(args.num_classes):
    data_list[j] = [i for i, label in enumerate(train_loader.dataset.targets) if label == j]

img_num_list = get_img_num_per_cls(args.dataset, args.imb_factor, args.num_meta * args.num_classes)
print(img_num_list)
print(sum(img_num_list))

im_data = {}
idx_to_del = []
for cls_idx, img_id_list in data_list.items():
    random.shuffle(img_id_list)
    img_num = img_num_list[int(cls_idx)]
    im_data[cls_idx] = img_id_list[img_num:]
    idx_to_del.extend(img_id_list[img_num:])

print(len(idx_to_del))
imbalanced_train_dataset = copy.deepcopy(train_data)
imbalanced_train_dataset.targets = np.delete(train_loader.dataset.targets, idx_to_del, axis=0)
imbalanced_train_dataset.data = np.delete(train_loader.dataset.data, idx_to_del, axis=0)
print(len(imbalanced_train_dataset))
imbalanced_train_loader = torch.utils.data.DataLoader(
    imbalanced_train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

validation_loader = torch.utils.data.DataLoader(
    train_data_meta, batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

best_prec1 = 0

# beta 是一个衰减因子
# img_num_list 是每个类别的样本数量列表
beta = 0.9999
# 计算了每个类别的有效样本数量
effective_num = 1.0 - np.power(beta, img_num_list)
# 计算了每个类别的权重
per_cls_weights = (1.0 - beta) / np.array(effective_num)
# 通过将权重标准化为总和为类别数量的值，得到了标准化后的权重。
per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(img_num_list)
# 将权重转换为 PyTorch 张量，并将其移动到 GPU 上，以便在训练过程中使用。
per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
# weights 是一个包含了每个类别权重的 PyTorch 张量。
# weights = torch.tensor(per_cls_weights).float()
weights = (per_cls_weights.clone().detach()).float()


def main():
    global args, best_prec1
    args = parser.parse_args()

    # 建立了一个ResNet32
    model = build_model()

    optimizer_a = torch.optim.SGD(model.params(), args.lr,
                                  momentum=args.momentum, nesterov=args.nesterov,
                                  weight_decay=args.weight_decay)

    cudnn.benchmark = True

    criterion = LDAM_meta(64, args.dataset == "cifar10" and 10 or 100, cls_num_list=img_num_list,
                          max_m=0.5, s=30)

    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer_a, epoch + 1)

        ratio = args.lam * float(epoch) / float(args.epochs)
        if epoch < 160:
            train(imbalanced_train_loader, model, optimizer_a, epoch)
        else:
            train_IDASAug(imbalanced_train_loader, validation_loader, model, optimizer_a, epoch, criterion, ratio)

        prec1, preds, gt_labels = validate(test_loader, model, nn.CrossEntropyLoss().cuda(), epoch)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        print('cur best acc:{}'.format(best_prec1))
        # 保存节点
        save_checkpoint(args, {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_prec1,
            'optimizer': optimizer_a.state_dict(),
        }, is_best, epoch)

    print('Best accuracy: ', best_prec1)
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = elapsed_time // 60
    print("总共跑了", minutes, "分钟")


def train(train_loader, model, optimizer_a, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    model.train()

    for i, (input, target) in enumerate(train_loader):

        target = torch.tensor(target, dtype=torch.long)
        target = target.cuda()

        # mixup
        input_var, target_a, target_b, lam = mixup_data(input, target)
        input_var, target_a, target_b = input_var.cuda(), target_a.cuda(), target_b.cuda()

        target_var = to_var(target, requires_grad=False)

        features, y_f = model(input_var, epoch)

        # Add
        l_f = mixup_criterion(y_f, target_a, target_b, lam)

        prec_train = accuracy(y_f.data, target_var.data, topk=(1,))[0]

        losses.update(l_f.item(), input.size(0))
        top1.update(prec_train.item(), input.size(0))

        optimizer_a.zero_grad()
        l_f.backward()
        optimizer_a.step()

        # target = torch.tensor(target, dtype=torch.long)
        # target = target.cuda()
        #
        # input_var = to_var(input, requires_grad=False)
        # target_var = to_var(target, requires_grad=False)
        #
        # features, y_f = model(input_var, epoch)
        # cost_w = F.cross_entropy(y_f, target_var, reduce=False)
        # l_f = torch.mean(cost_w)
        # prec_train = accuracy(y_f.data, target_var.data, topk=(1,))[0]
        #
        # losses.update(l_f.item(), input.size(0))
        # top1.update(prec_train.item(), input.size(0))
        #
        # optimizer_a.zero_grad()
        # l_f.backward()
        # optimizer_a.step()
        if i % args.print_freq == 0:
            print("--------------------------------Train------------------------------------")
            print('Epoch: [{0}]\t'
                  'Batch: [{1}/{2}]\t'
                  'Batch Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader),
                loss=losses, top1=top1))


def train_IDASAug(train_loader, validation_loader, model, optimizer_a, epoch, criterion, ratio):
    losses = AverageMeter()
    top1 = AverageMeter()
    model.train()

    for i, (input, target) in enumerate(train_loader):

        input_var = to_var(input, requires_grad=False)
        target_var = to_var(target, requires_grad=False)

        cv = criterion.get_cv()
        cv_var = to_var(cv)

        # meta
        meta_model = ResNet32_meta(args.dataset == 'cifar10' and 10 or 100)
        meta_model.load_state_dict(model.state_dict(), strict=False)
        meta_model.cuda()

        feat_hat, y_f_hat = meta_model(input_var, epoch)
        cls_loss_meta = criterion(meta_model.linear, feat_hat, y_f_hat, target_var, ratio,
                                  weights, cv_var, epoch, "none")
        meta_model.zero_grad()

        grads = torch.autograd.grad(cls_loss_meta, (meta_model.params()), create_graph=True)
        meta_lr = args.lr * ((0.01 ** int(epoch >= 160)) * (0.01 ** int(epoch >= 180)))
        meta_model.update_params(meta_lr, source_params=grads)

        input_val, target_val = next(iter(validation_loader))
        input_val_var = to_var(input_val, requires_grad=False)
        target_val_var = to_var(target_val, requires_grad=False)

        _, y_val = meta_model(input_val_var, epoch)
        cls_meta = F.cross_entropy(y_val, target_val_var)
        grad_cv = torch.autograd.grad(cls_meta, cv_var, only_inputs=True)[0]
        new_cv = cv_var - args.meta_lr * grad_cv

        del grad_cv, grads

        # main
        # model.train()
        features, predicts = model(input_var, epoch)
        cls_loss = criterion(model.linear, features, predicts, target_var, ratio, weights,
                             new_cv, epoch, "update")

        prec_train = accuracy(predicts.data, target_var.data, topk=(1,))[0]

        losses.update(cls_loss.item(), input.size(0))
        top1.update(prec_train.item(), input.size(0))

        optimizer_a.zero_grad()
        cls_loss.backward()
        optimizer_a.step()

        if i % args.print_freq == 0:
            print("---------------------------IDASAug Train------------------------------------")
            print('Epoch: [{0}]\t'
                  'Batch: [{1}/{2}]\t'
                  'Batch Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader),
                loss=losses, top1=top1))


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()

    # Add
    class_correct = [0] * 100
    class_total = [0] * 100

    true_labels = []
    preds = []

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        with torch.no_grad():
            _, output = model(input_var, epoch)

        output_numpy = output.data.cpu().numpy()
        preds_output = list(output_numpy.argmax(axis=1))

        true_labels += list(target_var.data.cpu().numpy())
        preds += preds_output

        prec1 = accuracy(output.data, target, topk=(1,))[0]
        top1.update(prec1.item(), input.size(0))

        # Add
        corrects = np.array(preds_output) == np.array(target_var.data.cpu().numpy())
        for i in range(len(target)):
            class_total[target[i]] += 1
            class_correct[target[i]] += corrects[i]
        # end Add

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print("---------------------------Begin Test--------------------------")
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1))

    print(' * Prec@1 {top1.avg:.3f}\t   Error:{Error:.3f}'.format(top1=top1, Error=(100 - top1.val)))
    print("---------------------------End All Test--------------------------")

    # Add
    for i in range(100):
        class_acc = class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        # print("Accuracy for class {}:{:.3f}%".format(i, class_acc * 100))
    print("---------------------------End Test--------------------------")


    return top1.avg, preds, true_labels


def build_model():
    model = ResNet32(args.dataset == 'cifar10' and 10 or 100)
    # model = ResNet32(num_classes=(10 if args.dataset == 'cifar10' else 100), use_modulated_att=True)

    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True

    return model


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * ((0.01 ** int(epoch >= 160)) * (0.01 ** int(epoch >= 180)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_cutime():
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_time = current_time.replace("-", "").replace(":", "").replace(" ", "")
    return current_time


def save_checkpoint(args, state, is_best, epoch):
    now_time = get_cutime()
    path = 'checkpoint/ours/0/' + str(now_time) + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    filename = path + str(epoch) + '_ckpt.pth.tar'
    if is_best:
        torch.save(state, filename)


def mixup_data(x, y, alpha=1.0):
    lam = torch.distributions.beta.Beta(alpha, alpha).sample().item()
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(pred, y_a, y_b, lam):
    return lam * F.cross_entropy(pred, y_a) + (1 - lam) * F.cross_entropy(pred, y_b)


def save_data(true_labels, preds):
    filename = 'data_at_epoch_200.txt'
    if not os.path.exists(filename):
        open(filename, 'w').close()

    with open(filename, 'w') as f:
        for true_labels, pred in zip(true_labels, preds):
            f.write(f'True Label:{true_labels}, Predicted Label:{pred}\n')
    print('Data saved at epoch 200.')


def save_class_accuracy(class_correct, class_total):
    filename = 'class_accuracy_at_epoch.txt'
    if not os.path.exists(filename):
        open(filename, 'w').close()

    with open(filename, 'w') as f:
        for i in range(100):
            class_acc = class_correct[i] / class_total[i] if class_total[i] > 0 else 0
            f.write(f'Class {i}:Accuracy {class_acc:.3f}\n')
        print('Class-wise accuracy saved at epoch 200')


if __name__ == '__main__':
    main()
