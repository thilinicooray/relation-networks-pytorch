import torch
from torch import nn
import torch.nn.functional as F
import utils
import torchvision as tv

class vgg_modified(nn.Module):
    def __init__(self):
        super(vgg_modified,self).__init__()
        self.vgg = tv.models.vgg16(pretrained=False)
        self.vgg_features = self.vgg.features
        #self.classifier = nn.Sequential(
        #nn.Dropout(),
        '''self.lin1 = nn.Linear(512 * 7 * 7, 1024)
        self.relu1 = nn.ReLU(True)
        self.dropout1 = nn.Dropout()
        self.lin2 =  nn.Linear(1024, 1024)
        self.relu2 = nn.ReLU(True)
        self.dropout2 = nn.Dropout()

        utils.initLinear(self.lin1)
        utils.initLinear(self.lin2)'''

    def rep_size(self): return 1024
    def base_size(self): return 512

    def forward(self,x):
        #return self.dropout2(self.relu2(self.lin2(self.dropout1(self.relu1(self.lin1(self.vgg_features(x).view(-1, 512*7*7)))))))
        return self.vgg_features(x)


class resnet_modified_small(nn.Module):
    def __init__(self):
        super(resnet_modified_small, self).__init__()
        self.resnet = tv.models.resnet34(pretrained=False)
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
            gpu_mode,
            conv_hidden=24,
            embed_hidden=300,
            lstm_hidden=300,
            mlp_hidden=256
    ):
        super().__init__()

        self.normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.train_transform = tv.transforms.Compose([
            tv.transforms.Resize(224),
            tv.transforms.RandomCrop(224),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            self.normalize,
        ])

        self.dev_transform = tv.transforms.Compose([
            tv.transforms.Resize(224),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            self.normalize,
        ])

        self.encoder = encoder
        self.gpu_mode = gpu_mode
        self.n_roles = self.encoder.get_num_roles()
        self.n_verbs = self.encoder.get_num_verbs()
        self.vocab_size = self.encoder.get_num_labels()
        self.max_role_count = self.encoder.get_max_role_count()

        self.conv = resnet_modified_small()


        self.verb = nn.Sequential(
            nn.Linear(7*7*self.conv.base_size(), mlp_hidden*2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(mlp_hidden*2, mlp_hidden*4),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(mlp_hidden*4, self.n_verbs),
        )


        '''self.verb = nn.Sequential(
            nn.Linear(7*7*self.conv.base_size(), mlp_hidden*2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(mlp_hidden*2, self.n_verbs),
        )'''
        '''self.verb = nn.Sequential(
            nn.Linear(7*7*self.conv.base_size(), self.n_verbs),
        )'''
        #self.verb.apply(utils.init_weight)

        '''self.role_lookup = nn.Embedding(self.n_roles+1, embed_hidden, padding_idx=self.n_roles)
        self.verb_lookup = nn.Embedding(self.n_verbs, embed_hidden)


        self.n_concat = self.conv.base_size() * 2 + embed_hidden + 2 * 2

        self.g = nn.Sequential(
            nn.Linear(self.n_concat, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
        )

        self.f = nn.Sequential(
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden*2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(mlp_hidden*2, self.vocab_size),
        )
        self.conv_hidden = self.conv.base_size()
        self.lstm_hidden = lstm_hidden
        self.mlp_hidden = mlp_hidden

        coords = torch.linspace(-4, 4, 7)
        x = coords.unsqueeze(0).repeat(7, 1)
        y = coords.unsqueeze(1).repeat(1, 7)
        coords = torch.stack([x, y]).unsqueeze(0)
        self.register_buffer('coords', coords)'''

    def train_preprocess(self): return self.train_transform
    def dev_preprocess(self): return self.dev_transform

    def forward(self, image):

        '''print('testing 123')
        x = torch.tensor([[1, 2, 3],[4,5,6]])
        print('original', x.size())
        x = x.repeat(1,2)
        print('xxxxxx', x, x.view(-1,3), x.size())'''

        conv = self.conv(image)

        #verb pred
        verb_pred = self.verb(conv.view(-1, 7*7*self.conv.base_size()))



        return verb_pred





    def calculate_loss(self, verb_pred, gt_verbs,gt_labels):

        batch_size = verb_pred.size()[0]
        loss = 0
        #print('eval pred verbs :', pred_verbs)
        for i in range(batch_size):
            verb_loss = utils.cross_entropy_loss(verb_pred[i], gt_verbs[i])
            loss += verb_loss
            '''for index in range(gt_labels.size()[1]):
                frame_loss = 0
                verb_loss = utils.cross_entropy_loss(verb_pred[i], gt_verbs[i])


                #frame_loss += verb_loss
                #print('frame loss', frame_loss)
                loss += verb_loss'''


        final_loss = loss/batch_size
        #print('loss :', final_loss)
        return final_loss

    def calculate_eval_loss(self, verb_pred, gt_verbs, gt_labels):

        batch_size = verb_pred.size()[0]
        loss = 0
        #print('eval pred verbs :', pred_verbs)
        for i in range(batch_size):
            for index in range(gt_labels.size()[1]):
                frame_loss = 0
                verb_loss = utils.cross_entropy_loss(verb_pred[i], gt_verbs[i])


                #frame_loss += verb_loss
                #print('frame loss', frame_loss)
                loss += verb_loss


        final_loss = loss/batch_size
        #print('loss :', final_loss)
        return final_loss

