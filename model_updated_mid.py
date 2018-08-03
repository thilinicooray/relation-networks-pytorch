import torch
from torch import nn
from torch.nn.init import kaiming_uniform_, normal_
import torch.nn.functional as F
import utils
import torchvision as tv

class resnet_modified_small(nn.Module):
    def __init__(self):
        super(resnet_modified_small, self).__init__()
        self.resnet = tv.models.resnet34(pretrained=True)
        self.dropout2d = nn.Dropout2d(.5)
        #probably want linear, relu, dropout
        '''self.linear = nn.Linear(7*7*512, 1024)
        self.dropout2d = nn.Dropout2d(.5)
        self.dropout = nn.Dropout(.5)
        self.relu = nn.LeakyReLU()
        utils.init_weight(self.linear)'''

    def base_size(self): return 512
    def rep_size(self): return 1024

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.dropout2d(x)

        #return self.dropout(self.relu(self.linear(x.view(-1, 7*7*self.base_size()))))
        return x


class RelationNetworks(nn.Module):
    def __init__(
            self,
            encoder,
            conv_hidden=24,
            embed_hidden=256,
            lstm_hidden=256,
            mlp_hidden=512
    ):
        super().__init__()

        self.normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.train_transform = tv.transforms.Compose([
            tv.transforms.Resize(128),
            tv.transforms.Pad(8),
            tv.transforms.RandomCrop([128, 128]),
            tv.transforms.RandomRotation(2.8),
            tv.transforms.ToTensor(),
            self.normalize,
        ])

        self.dev_transform = tv.transforms.Compose([
            tv.transforms.Resize(128),
            tv.transforms.ToTensor(),
            self.normalize,
        ])

        self.encoder = encoder
        self.n_roles = self.encoder.get_num_roles()
        self.n_verbs = self.encoder.get_num_verbs()
        self.vocab_size = self.encoder.get_num_labels()
        self.max_role_count = self.encoder.get_max_role_count()

        #self.conv = resnet_modified_small()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 24, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 24, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 24, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(),
        )

        self.verb = nn.Sequential(
            nn.Linear(24*8*8, mlp_hidden),
            nn.ReLU(),
            #nn.Linear(mlp_hidden, mlp_hidden),
            #nn.ReLU(),
            nn.Dropout(),
            nn.Linear(mlp_hidden, self.n_verbs),
        )

        self.role_lookup = nn.Embedding(self.n_roles+1, embed_hidden, padding_idx=self.n_roles)
        self.verb_lookup = nn.Embedding(self.n_verbs, embed_hidden)


        self.n_concat = 24 * 2 + embed_hidden + 2 * 2

        self.g = nn.Sequential(
            nn.Linear(self.n_concat, mlp_hidden),
            #nn.ReLU(),
            #nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
        )

        self.f = nn.Sequential(
            #nn.Linear(mlp_hidden, mlp_hidden),
            #nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden*4),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(mlp_hidden*4, self.vocab_size),
        )

        self.conv_hidden = 24
        self.lstm_hidden = lstm_hidden
        self.mlp_hidden = mlp_hidden

        coords = torch.linspace(-4, 4, 8)
        x = coords.unsqueeze(0).repeat(8, 1)
        y = coords.unsqueeze(1).repeat(1, 8)
        coords = torch.stack([x, y]).unsqueeze(0)
        self.register_buffer('coords', coords)

    def train_preprocess(self): return self.train_transform
    def dev_preprocess(self): return self.dev_transform

    def forward(self, image, verbs, roles):

        '''print('testing 123')
        x = torch.tensor([[1, 2, 3],[4,5,6]])
        print('original', x.size())
        x = x.repeat(1,2)
        print('xxxxxx', x, x.view(-1,3), x.size())'''

        conv = self.conv(image)

        #verb pred
        verb_pred = self.verb(conv.view(-1, 24*8*8))

        batch_size, n_channel, conv_h, conv_w = conv.size()
        n_pair = conv_h * conv_w

        verb_embd = self.verb_lookup(verbs)
        role_embd = self.role_lookup(roles)

        role_embed_reshaped = role_embd.transpose(0,1)
        verb_embed_expand = verb_embd.expand(self.max_role_count, verb_embd.size(0), verb_embd.size(1))
        role_verb_embd = verb_embed_expand * role_embed_reshaped
        role_verb_embd =  role_verb_embd.transpose(0,1)
        role_verb_embd = role_verb_embd.contiguous().view(-1, self.lstm_hidden)
        #new batch size = batch_size*max_role
        batch_size_updated = role_verb_embd.size(0)
        print('new', batch_size_updated, n_pair, role_verb_embd.size())

        qst = torch.unsqueeze(role_verb_embd, 1)
        qst = qst.repeat(1,n_pair * n_pair,1)
        qst = torch.squeeze(qst)

        print('qst size', qst.size())

        '''h_tile = role_verb_embd.permute(1, 0, 2).expand(
            batch_size_updated, n_pair * n_pair, self.lstm_hidden
        )'''


        #update conv to expand for all roles in 1 image
        conv = conv.repeat(1,self.max_role_count, 1, 1)
        #print('conv, size', conv.size())
        conv = conv.view(-1, n_channel, conv_h, conv_w)
        print('after view', conv.size(), self.coords.size())
        conv = torch.cat([conv, self.coords.expand(batch_size_updated, 2, conv_h, conv_w)], 1)
        n_channel += 2
        conv_tr = conv.view(batch_size_updated, n_channel, -1).permute(0, 2, 1)
        conv1 = conv_tr.unsqueeze(1).expand(batch_size_updated, n_pair, n_pair, n_channel)
        conv2 = conv_tr.unsqueeze(2).expand(batch_size_updated, n_pair, n_pair, n_channel)
        conv1 = conv1.contiguous().view(-1, n_pair * n_pair, n_channel)
        conv2 = conv2.contiguous().view(-1, n_pair * n_pair, n_channel)
        #print('size :', conv2.size())
        concat_vec = torch.cat([conv1, conv2, qst], 2).view(-1, self.n_concat)
        g = self.g(concat_vec)
        g = g.view(-1, n_pair * n_pair, self.mlp_hidden).sum(1).squeeze()

        f = self.f(g)

        role_predict = f.contiguous().view(batch_size, -1, self.vocab_size)
        #print('ffffff', f.size())

        return verb_pred, role_predict

    def calculate_loss(self, verb_pred, gt_verbs, role_label_pred, gt_labels):

        batch_size = verb_pred.size()[0]
        loss = 0
        for i in range(batch_size):
            for index in range(gt_labels.size()[1]):
                frame_loss = 0
                verb_loss = utils.cross_entropy_loss(verb_pred[i], gt_verbs[i])
                for j in range(0, self.max_role_count):
                    frame_loss += utils.cross_entropy_loss(role_label_pred[i][j], gt_labels[i,index,j] ,self.vocab_size)
                frame_loss = verb_loss + frame_loss/len(self.encoder.verb2_role_dict[self.encoder.verb_list[gt_verbs[i]]])
                #print('frame loss', frame_loss)
                loss += frame_loss


        final_loss = loss/batch_size
        #print('loss :', final_loss)
        return final_loss
