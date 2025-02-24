import torch
import os
import torch.nn.functional as F
import torchvision.utils as vutils

class Criterion(object):
    def __init__(self, args):
        self.setupNormalCrit(args)

    def setupNormalCrit(self, args):
        print('=> Using {} for criterion normal'.format(args.normal_loss))
        self.normal_loss = args.normal_loss
        self.normal_w = args.normal_w
        if args.normal_loss == 'mse':
            self.n_crit = torch.nn.MSELoss()
        elif args.normal_loss == 'cos':
            self.n_crit = torch.nn.CosineEmbeddingLoss()
            self.att_crit1 = torch.nn.L1Loss(reduce=False, size_average=False)
        else:
            raise Exception("=> Unknown Criterion '{}'".format(args.normal_loss))
        if args.cuda:
            self.n_crit = self.n_crit.cuda()

    def forward(self, output1, output2, output3, target):
        # output3
        self.h_x = output3.size()[2]
        self.w_x = output3.size()[3]
        self.r = F.pad(output3, (0, 1, 0, 0))[:, :, :, 1:]
        self.l = F.pad(output3, (1, 0, 0, 0))[:, :, :, :self.w_x]
        self.t = F.pad(output3, (0, 0, 1, 0))[:, :, :self.h_x, :]
        self.b = F.pad(output3, (0, 0, 0, 1))[:, :, 1:, :]
        self.outputgrad_x = torch.pow((self.r - self.l) * 0.5, 2).permute(0, 2, 3, 1).contiguous().view(-1, 3)
        self.outputgrad_y = torch.pow((self.t - self.b) * 0.5, 2).permute(0, 2, 3, 1).contiguous().view(-1, 3)

        # ground-truth
        self.h_x1 = target.size()[2]
        self.w_x1 = target.size()[3]
        self.r1 = F.pad(target, (0, 1, 0, 0))[:, :, :, 1:]
        self.l1 = F.pad(target, (1, 0, 0, 0))[:, :, :, :self.w_x1]
        self.t1 = F.pad(target, (0, 0, 1, 0))[:, :, :self.h_x1, :]
        self.b1 = F.pad(target, (0, 0, 0, 1))[:, :, 1:, :]
        self.targetgrad_x = torch.pow((self.r1 - self.l1) * 0.5, 2).permute(0, 2, 3, 1).contiguous().view(-1, 3)
        self.targetgrad_y = torch.pow((self.t1 - self.b1) * 0.5, 2).permute(0, 2, 3, 1).contiguous().view(-1, 3)

        # output2
        self.h_x2 = output2.size()[2]
        self.w_x2 = output2.size()[3]
        self.r2 = F.pad(output2, (0, 1, 0, 0))[:, :, :, 1:]
        self.l2 = F.pad(output2, (1, 0, 0, 0))[:, :, :, :self.w_x2]
        self.t2 = F.pad(output2, (0, 0, 1, 0))[:, :, :self.h_x2, :]
        self.b2 = F.pad(output2, (0, 0, 0, 1))[:, :, 1:, :]
        self.outputgrad_x2 = torch.pow((self.r2 - self.l2) * 0.5, 2).permute(0, 2, 3, 1).contiguous().view(-1, 3)
        self.outputgrad_y2 = torch.pow((self.t2 - self.b2) * 0.5, 2).permute(0, 2, 3, 1).contiguous().view(-1, 3)

        # output1
        self.h_x3 = output1.size()[2]
        self.w_x3 = output1.size()[3]
        self.r3 = F.pad(output1, (0, 1, 0, 0))[:, :, :, 1:]
        self.l3 = F.pad(output1, (1, 0, 0, 0))[:, :, :, :self.w_x3]
        self.t3 = F.pad(output1, (0, 0, 1, 0))[:, :, :self.h_x3, :]
        self.b3 = F.pad(output1, (0, 0, 0, 1))[:, :, 1:, :]
        self.outputgrad_x3 = torch.pow((self.r3 - self.l3) * 0.5, 2).permute(0, 2, 3, 1).contiguous().view(-1, 3)
        self.outputgrad_y3 = torch.pow((self.t3 - self.b3) * 0.5, 2).permute(0, 2, 3, 1).contiguous().view(-1, 3)

        if self.normal_loss == 'cos':
            num = target.nelement() // target.shape[1]
            if not hasattr(self, 'flag') or num != self.flag.nelement():
                self.flag = torch.autograd.Variable(target.data.new().resize_(num).fill_(1))

            self.out1_reshape = output1.permute(0, 2, 3, 1).contiguous().view(-1, 3)
            self.out2_reshape = output2.permute(0, 2, 3, 1).contiguous().view(-1, 3)
            self.out3_reshape = output3.permute(0, 2, 3, 1).contiguous().view(-1, 3)
            self.gt_reshape = target.permute(0, 2, 3, 1).contiguous().view(-1, 3)
            self.loss1 = self.n_crit(self.out1_reshape, self.gt_reshape, self.flag)
            self.loss2 = self.n_crit(self.out2_reshape, self.gt_reshape, self.flag)
            self.loss3 = self.n_crit(self.out3_reshape, self.gt_reshape, self.flag)
            self.loss = 0.5 * self.loss1 + 0.7 * self.loss2 + 1.0 * self.loss3
            self.gloss1 = torch.mean(self.att_crit1(self.outputgrad_x3, self.targetgrad_x)) + torch.mean(self.att_crit1(self.outputgrad_y3, self.targetgrad_y))
            self.gloss2 = torch.mean(self.att_crit1(self.outputgrad_x2, self.targetgrad_x)) + torch.mean(self.att_crit1(self.outputgrad_y2, self.targetgrad_y))
            self.gloss3 = torch.mean(self.att_crit1(self.outputgrad_x, self.targetgrad_x)) + torch.mean(self.att_crit1(self.outputgrad_y,self.targetgrad_y))
            self.gloss = 0.5 * self.gloss1 + 0.7 * self.gloss2 + 1.0 * self.gloss3
            self.loss = self.loss + self.gloss * 0.05
        elif self.normal_loss == 'mse':
            self.loss = self.normal_w * self.n_crit(output, target)
        out_loss = {'N_loss': self.loss.item(), 'N_loss1': self.loss1.item(), 'N_loss2': self.loss2.item(), 'N_loss3': self.loss3.item(),
                    'gloss': self.gloss.item(), 'gloss_1': self.gloss1.item(), 'gloss_2': self.gloss2.item(), 'gloss_3': self.gloss3.item()}
        return out_loss

    def backward(self):
        self.loss.backward()


def getOptimizer(args, params):
    print('=> Using %s solver for optimization' % (args.solver))
    if args.solver == 'adam':
        optimizer = torch.optim.Adam(params, args.init_lr, betas=(args.beta_1, args.beta_2))
    elif args.solver == 'sgd':
        optimizer = torch.optim.SGD(params, args.init_lr, momentum=args.momentum)
    else:
        raise Exception("=> Unknown Optimizer %s" % (args.solver))
    return optimizer


def getLrScheduler(args, optimizer):
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_decay,
                                                     last_epoch=args.start_epoch - 2)
    return scheduler


def loadRecords(path, model, optimizer):
    records = None
    if os.path.isfile(path):
        records = torch.load(path[:-8] + '_rec' + path[-8:])
        optimizer.load_state_dict(records['optimizer'])
        start_epoch = records['epoch'] + 1
        records = records['records']
        print("=> loaded Records")
    else:
        raise Exception("=> no checkpoint found at '{}'".format(path))
    return records, start_epoch


def configOptimizer(args, model):
    records = None
    optimizer = getOptimizer(args, model.parameters())
    if args.resume:
        print("=> Resume loading checkpoint '{}'".format(args.resume))
        records, start_epoch = loadRecords(args.resume, model, optimizer)
        args.start_epoch = start_epoch
    scheduler = getLrScheduler(args, optimizer)
    return optimizer, scheduler, records
