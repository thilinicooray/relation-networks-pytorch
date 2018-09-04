import torch
from torch import nn
import torch.nn.functional as F
import utils
import torchvision as tv
import math
import copy

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

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    #print('scores :', scores)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    #print('att ', p_attn)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class FeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        #only 1 linear layer
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        #self.linears = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.size = d_model

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        #what is happening here??????
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class DecoderLayer(nn.Module):
    "Decoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward= feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Role_Labeller(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Role_Labeller, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)

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
            nn.Linear(mlp_hidden*2, mlp_hidden*4),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(mlp_hidden*4, self.n_verbs),
        )

        self.role_lookup = nn.Embedding(self.n_roles+1, embed_hidden, padding_idx=self.n_roles)
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

        self.f_1 = nn.Sequential(
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
        )

        c = copy.deepcopy
        attn = MultiHeadedAttention(h=2, d_model=mlp_hidden)
        ff = FeedForward(mlp_hidden, d_ff=mlp_hidden*2, dropout=0.1)

        self.f = Role_Labeller(DecoderLayer(mlp_hidden, c(attn),c(ff), 0.1), 3)
        for p in self.f.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden*2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(mlp_hidden*2, self.vocab_size+1),
        )

        self.conv_hidden = self.conv.base_size()
        self.lstm_hidden = lstm_hidden
        self.mlp_hidden = mlp_hidden

        coords = torch.linspace(-4, 4, 7)
        x = coords.unsqueeze(0).repeat(7, 1)
        y = coords.unsqueeze(1).repeat(1, 7)
        coords = torch.stack([x, y]).unsqueeze(0)
        self.register_buffer('coords', coords)

    def train_preprocess(self): return self.train_transform
    def dev_preprocess(self): return self.dev_transform

    def forward(self, image, verbs, roles):
        #print('came here')

        '''print('testing 123')
        x = torch.tensor([[1, 2, 3],[4,5,6]])
        print('original', x.size())
        x = x.repeat(1,2)
        print('xxxxxx', x, x.view(-1,3), x.size())'''

        conv = self.conv(image)

        #verb pred
        verb_pred = self.verb(conv.view(-1, 7*7*self.conv.base_size()))

        batch_size, n_channel, conv_h, conv_w = conv.size()
        n_pair = conv_h * conv_w

        verb_embd = self.verb_lookup(verbs)
        role_embd = self.role_lookup(roles)
        #print('verb embed :', verb_embd.size())
        #print('role embed :', role_embd)

        role_embed_reshaped = role_embd.transpose(0,1)
        verb_embed_expand = verb_embd.expand(self.max_role_count, verb_embd.size(0), verb_embd.size(1))
        role_verb_embd = verb_embed_expand * role_embed_reshaped
        role_verb_embd =  role_verb_embd.transpose(0,1)
        role_verb_embd = role_verb_embd.contiguous().view(-1, self.lstm_hidden)
        #new batch size = batch_size*max_role
        #print("role_verb_embd" , role_verb_embd)
        batch_size_updated = role_verb_embd.size(0)
        #print('new', batch_size_updated, n_pair, role_verb_embd.size())

        qst = torch.unsqueeze(role_verb_embd, 1)
        qst = qst.repeat(1,n_pair * n_pair,1)
        qst = torch.squeeze(qst)

        #print('qst size', qst.size())

        '''h_tile = role_verb_embd.permute(1, 0, 2).expand(
            batch_size_updated, n_pair * n_pair, self.lstm_hidden
        )'''


        #update conv to expand for all roles in 1 image
        conv = conv.repeat(1,self.max_role_count, 1, 1)
        #print('conv, size', conv.size())
        conv = conv.view(-1, n_channel, conv_h, conv_w)
        #print('after view', conv.size())
        conv = torch.cat([conv, self.coords.expand(batch_size_updated, 2, conv_h, conv_w)], 1)
        n_channel += 2
        conv_tr = conv.view(batch_size_updated, n_channel, -1).permute(0, 2, 1)
        conv1 = conv_tr.unsqueeze(1).expand(batch_size_updated, n_pair, n_pair, n_channel)
        conv2 = conv_tr.unsqueeze(2).expand(batch_size_updated, n_pair, n_pair, n_channel)
        conv1 = conv1.contiguous().view(-1, n_pair * n_pair, n_channel)
        conv2 = conv2.contiguous().view(-1, n_pair * n_pair, n_channel)
        #print('size :', conv2.size())
        #print('no issue efore cat')
        concat_vec = torch.cat([conv1, conv2, qst], 2).view(-1, self.n_concat)
        #print('no issue after cat')
        g = self.g(concat_vec)

        '''if self.gpu_mode >= 0:
            torch.cuda.empty_cache()'''
        #print('no issue after g')
        g = g.view(-1, n_pair * n_pair, self.mlp_hidden).sum(1).squeeze()
        f1 = self.f_1(g)
        f1 = f1.contiguous().view(batch_size, -1, self.mlp_hidden)
        #print('g out size :', g.size())
        #print('no issue after g view')
        mask = self.encoder.get_adj_matrix(verbs)
        if self.gpu_mode >= 0:
            mask = mask.to(torch.device('cuda'))
        #print('mask ', mask.size(), mask)
        f = self.f(f1, mask)
        f = self.classifier(f)
        #print('no issue after f')
        '''if self.gpu_mode >= 0:
            torch.cuda.empty_cache()'''

        role_predict = f.contiguous().view(batch_size, -1, self.vocab_size+1)
        #print('ffffff', f.size())

        #del f, g

        return verb_pred, role_predict

    def forward_eval(self, image):
        conv = self.conv(image)

        #verb pred
        verb_pred = self.verb(conv.view(-1, 7*7*self.conv.base_size()))
        #print('verb pred', verb_pred.size())
        sorted_idx = torch.sort(verb_pred, 1, True)[1]
        #print('sorted ', sorted_idx.size())
        verbs = sorted_idx[:,0]
        #todo:change for 5
        #print('top1 verbs', verbs)

        #print('verbs :', verbs.size(), verbs)

        roles = self.encoder.get_role_ids_batch(verbs)

        roles = roles.type(torch.LongTensor)
        verbs = verbs.type(torch.LongTensor)

        if self.gpu_mode >= 0:
            roles = roles.to(torch.device('cuda'))
            verbs = verbs.to(torch.device('cuda'))

        batch_size, n_channel, conv_h, conv_w = conv.size()
        n_pair = conv_h * conv_w

        verb_embd = self.verb_lookup(verbs)
        #print('verb embed :', verb_embd.size())
        role_embd = self.role_lookup(roles)
        #print('role embed :', role_embd.size())

        role_embed_reshaped = role_embd.transpose(0,1)
        verb_embed_expand = verb_embd.expand(self.max_role_count, verb_embd.size(0), verb_embd.size(1))
        role_verb_embd = verb_embed_expand * role_embed_reshaped
        role_verb_embd =  role_verb_embd.transpose(0,1)
        role_verb_embd = role_verb_embd.contiguous().view(-1, self.lstm_hidden)
        #new batch size = batch_size*max_role
        batch_size_updated = role_verb_embd.size(0)
        #print('new', batch_size_updated, n_pair, role_verb_embd.size())

        qst = torch.unsqueeze(role_verb_embd, 1)
        qst = qst.repeat(1,n_pair * n_pair,1)
        qst = torch.squeeze(qst)

        #print('qst size', qst.size())

        '''h_tile = role_verb_embd.permute(1, 0, 2).expand(
            batch_size_updated, n_pair * n_pair, self.lstm_hidden
        )'''


        #update conv to expand for all roles in 1 image
        conv = conv.repeat(1,self.max_role_count, 1, 1)
        #print('conv, size', conv.size())
        conv = conv.view(-1, n_channel, conv_h, conv_w)
        #print('after view', conv.size())
        conv = torch.cat([conv, self.coords.expand(batch_size_updated, 2, conv_h, conv_w)], 1)
        n_channel += 2
        conv_tr = conv.view(batch_size_updated, n_channel, -1).permute(0, 2, 1)
        conv1 = conv_tr.unsqueeze(1).expand(batch_size_updated, n_pair, n_pair, n_channel)
        conv2 = conv_tr.unsqueeze(2).expand(batch_size_updated, n_pair, n_pair, n_channel)
        conv1 = conv1.contiguous().view(-1, n_pair * n_pair, n_channel)
        conv2 = conv2.contiguous().view(-1, n_pair * n_pair, n_channel)
        #print('size :', conv2.size())
        #print('no issue efore cat')
        concat_vec = torch.cat([conv1, conv2, qst], 2).view(-1, self.n_concat)
        #print('no issue after cat')
        g = self.g(concat_vec)

        '''if self.gpu_mode >= 0:
            torch.cuda.empty_cache()'''
        #print('no issue after g')
        g = g.view(-1, n_pair * n_pair, self.mlp_hidden).sum(1).squeeze()
        f1 = self.f_1(g)
        f1 = f1.contiguous().view(batch_size, -1, self.mlp_hidden)
        #print('g out size :', g.size())
        #print('no issue after g view')
        mask = self.encoder.get_adj_matrix(verbs)
        if self.gpu_mode >= 0:
            mask = mask.to(torch.device('cuda'))
        #print('mask ', mask.size())
        f = self.f(f1, mask)
        f = self.classifier(f)
        #print('no issue after f')
        '''if self.gpu_mode >= 0:
            torch.cuda.empty_cache()'''

        role_predict = f.contiguous().view(batch_size, -1, self.vocab_size+1)
        #print('ffffff', f.size())

        #del f, g

        return verb_pred, role_predict





    def calculate_loss(self, verb_pred, gt_verbs, role_label_pred, gt_labels,args):

        batch_size = verb_pred.size()[0]
        if args.train_all:
            loss = 0
            for i in range(batch_size):
                for index in range(gt_labels.size()[1]):
                    frame_loss = 0
                    verb_loss = utils.cross_entropy_loss(verb_pred[i], gt_verbs[i])
                    #frame_loss = criterion(role_label_pred[i], gt_labels[i,index])
                    for j in range(0, self.max_role_count):
                        frame_loss += utils.cross_entropy_loss(role_label_pred[i][j], gt_labels[i,index,j] ,self.vocab_size)
                    frame_loss = verb_loss + frame_loss/len(self.encoder.verb2_role_dict[self.encoder.verb_list[gt_verbs[i]]])
                    #print('frame loss', frame_loss, 'verb loss', verb_loss)
                    loss += frame_loss
        else:
            #verb from pre-trained
            loss = 0
            for i in range(batch_size):
                for index in range(gt_labels.size()[1]):
                    frame_loss = 0
                    #verb_loss = utils.cross_entropy_loss(verb_pred[i], gt_verbs[i])
                    #frame_loss = criterion(role_label_pred[i], gt_labels[i,index])
                    for j in range(0, self.max_role_count):
                        frame_loss += utils.cross_entropy_loss(role_label_pred[i][j], gt_labels[i,index,j] ,self.vocab_size)
                    frame_loss = frame_loss/len(self.encoder.verb2_role_dict[self.encoder.verb_list[gt_verbs[i]]])
                    #print('frame loss', frame_loss, 'verb loss', verb_loss)
                    loss += frame_loss


        final_loss = loss/batch_size
        #print('loss :', final_loss)
        return final_loss

    def calculate_eval_loss(self, verb_pred, gt_verbs, role_label_pred, gt_labels,args):

        batch_size = verb_pred.size()[0]

        sorted_idx = torch.sort(verb_pred, 1, True)[1]
        pred_verbs = sorted_idx[:,0]
        #print('eval pred verbs :', pred_verbs)
        if args.train_all:
            loss = 0
            for i in range(batch_size):
                for index in range(gt_labels.size()[1]):
                    frame_loss = 0
                    verb_loss = utils.cross_entropy_loss(verb_pred[i], gt_verbs[i])
                    gt_role_list = self.encoder.get_role_ids(gt_verbs[i])
                    pred_role_list = self.encoder.get_role_ids(pred_verbs[i])

                    #print ('role list diff :', gt_role_list, pred_role_list)

                    for j in range(0, self.max_role_count):
                        if pred_role_list[j] == len(self.encoder.role_list):
                            continue
                        if pred_role_list[j] in gt_role_list:
                            #print('eval loss :', gt_role_list, pred_role_list[j])
                            g_idx = (gt_role_list == pred_role_list[j]).nonzero()
                            #print('found idx' , g_idx)
                            frame_loss += utils.cross_entropy_loss(role_label_pred[i][j], gt_labels[i,index,g_idx] ,self.vocab_size)

                    frame_loss = verb_loss + frame_loss/len(self.encoder.verb2_role_dict[self.encoder.verb_list[gt_verbs[i]]])
                    #print('frame loss', frame_loss)
                    loss += frame_loss
        else:
            loss = 0
            for i in range(batch_size):
                for index in range(gt_labels.size()[1]):
                    frame_loss = 0
                    verb_loss = utils.cross_entropy_loss(verb_pred[i], gt_verbs[i])
                    gt_role_list = self.encoder.get_role_ids(gt_verbs[i])
                    pred_role_list = self.encoder.get_role_ids(pred_verbs[i])

                    #print ('role list diff :', gt_role_list, pred_role_list)

                    for j in range(0, self.max_role_count):
                        if pred_role_list[j] == len(self.encoder.role_list):
                            continue
                        if pred_role_list[j] in gt_role_list:
                            #print('eval loss :', gt_role_list, pred_role_list[j])
                            g_idx = (gt_role_list == pred_role_list[j]).nonzero()
                            #print('found idx' , g_idx)
                            frame_loss += utils.cross_entropy_loss(role_label_pred[i][j], gt_labels[i,index,g_idx] ,self.vocab_size)

                    frame_loss = frame_loss/len(self.encoder.verb2_role_dict[self.encoder.verb_list[gt_verbs[i]]])
                    #print('frame loss', frame_loss)
                    loss += frame_loss


        final_loss = loss/batch_size
        #print('loss :', final_loss)
        return final_loss
