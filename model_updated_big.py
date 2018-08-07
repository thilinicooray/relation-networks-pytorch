import torch
from torch import nn
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
            gpu_mode,
            conv_hidden=24,
            embed_hidden=512,
            lstm_hidden=512,
            mlp_hidden=1024
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
            nn.Linear(7*7*self.conv.base_size(), mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(mlp_hidden, self.n_verbs),
        )

        self.role_lookup = nn.Embedding(self.n_roles+1, embed_hidden, padding_idx=self.n_roles)
        self.verb_lookup = nn.Embedding(self.n_verbs, embed_hidden)


        self.n_concat = self.conv.base_size() * 2 + embed_hidden + 2 * 2

        self.g = nn.Sequential(
            nn.Linear(self.n_concat, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden*2),
            nn.ReLU(),
            nn.Linear(mlp_hidden*2, mlp_hidden*4),
            nn.ReLU(),
        )

        self.f = nn.Sequential(
            nn.Linear(mlp_hidden*4, mlp_hidden*4),
            nn.ReLU(),
            nn.Linear(mlp_hidden*4, mlp_hidden*2),
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
        self.register_buffer('coords', coords)

    def train_preprocess(self): return self.train_transform
    def dev_preprocess(self): return self.dev_transform

    def forward1(self, image, verbs, roles):

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
        g = g.view(-1, n_pair * n_pair, self.mlp_hidden*4).sum(1).squeeze()
        #print('no issue after g view')
        f = self.f(g)
        #print('no issue after f')
        '''if self.gpu_mode >= 0:
            torch.cuda.empty_cache()'''

        role_predict = f.contiguous().view(batch_size, -1, self.vocab_size)
        #print('ffffff', f.size())

        #del f, g

        return verb_pred, role_predict

    #def forward_eval(self, image):
    def forward(self, image):
        conv = self.conv(image)

        #verb pred
        verb_pred = self.verb(conv.view(-1, 7*7*self.conv.base_size()))
        #print('verb pred', verb_pred.size())
        sorted_idx = torch.sort(verb_pred, 1, True)[1]
        #print('sorted ', sorted_idx.size())
        verbs = sorted_idx[:,0]

        #print('verbs :', verbs.size(), verbs)

        #roles = self.encoder.get_role_ids_batch(verbs)
        roles = torch.arange(0, self.n_roles)

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
        verb_embed_expand = verb_embd.expand(self.n_roles, verb_embd.size(0), verb_embd.size(1))
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
        conv = conv.repeat(1,self.n_roles, 1, 1)
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
        print('full :', concat_vec.size())
        g = self.g(concat_vec)

        '''if self.gpu_mode >= 0:
            torch.cuda.empty_cache()'''
        #print('no issue after g')
        g = g.view(-1, n_pair * n_pair, self.mlp_hidden*4).sum(1).squeeze()
        #print('no issue after g view')
        f = self.f(g)
        #print('no issue after f')
        '''if self.gpu_mode >= 0:
            torch.cuda.empty_cache()'''

        role_predict = f.contiguous().view(batch_size, -1, self.vocab_size)
        #print('ffffff', f.size())

        #del f, g

        return verb_pred, role_predict

    def forward_eval_beam(self, image, beam=5):
        conv_org = self.conv(image)



        #verb pred
        verb_pred = self.verb(conv_org.view(-1, 7*7*self.conv.base_size()))
        logsm = nn.LogSoftmax()
        verb_pred_prob = logsm(verb_pred)
        #print('verb pred', verb_pred.size())
        sorted_prob, sorted_idx = torch.sort(verb_pred_prob, 1, True)
        #print('sorted ', sorted_idx.size())
        verbs = sorted_idx[:, 0:beam]
        beam_verb_prob = sorted_prob[:, 0:beam]

        #print('verb sizes ', verbs.size(), beam_verb_prob.size())

        #print('verbs :', verbs.size(), verbs)

        for b in range(0,beam):
            conv = conv_org

            roles = self.encoder.get_role_ids_batch(verbs[:,b])

            roles = roles.type(torch.LongTensor)
            verbs = verbs.type(torch.LongTensor)

            if self.gpu_mode >= 0:
                roles = roles.to(torch.device('cuda'))
                verbs = verbs.to(torch.device('cuda'))

            batch_size, n_channel, conv_h, conv_w = conv.size()
            n_pair = conv_h * conv_w

            verb_embd = self.verb_lookup(verbs[:,b])
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
            #print('conv sizeeee', conv.size())
            conv = conv.repeat(1,self.max_role_count, 1, 1)
            #print('conv, size', conv.size())
            conv = conv.view(-1, n_channel, conv_h, conv_w)
            #print('after view', conv.size(), conv_h, conv_w)
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
            g = g.view(-1, n_pair * n_pair, self.mlp_hidden*4).sum(1).squeeze()
            #print('no issue after g view')
            f = self.f(g)
            #print('no issue after f')
            '''if self.gpu_mode >= 0:
                torch.cuda.empty_cache()'''

            role_predict = f.contiguous().view(batch_size, -1, self.vocab_size)
            role_predict_prob = logsm(role_predict)
            role_max_prob, role_max_idx = torch.max(role_predict_prob, -1) #batch_sizex6
            #print('role max prob :', role_max_prob.size())
            all_situation_prob = torch.cat((torch.unsqueeze(beam_verb_prob[:,b],1), role_max_prob),1)
            situation_joint_prob = torch.sum(all_situation_prob,1)

            if b == 0:
                beam_role_idx = role_max_idx
                beam_joint_prob = torch.unsqueeze(situation_joint_prob,1)
            else:
                #print('check' , beam_role_idx.size(), role_max_idx.size(), beam_joint_prob.size(), situation_joint_prob.size())
                beam_role_idx = torch.cat((beam_role_idx.clone(), role_max_idx), 1)
                beam_joint_prob = torch.cat((beam_joint_prob.clone(), torch.unsqueeze(situation_joint_prob,1)), 1)

        #print('sizes of beam loaders', verbs.size(), beam_role_idx.size(), beam_joint_prob.size())

        sorted_prob, sorted_joint_idx = torch.sort(beam_joint_prob, 1, True)
        #print('joint prob ', beam_joint_prob)
        sorted_beam_verb_ids = []
        max_beam_role_label_ids = []

        for i in range(0,batch_size):
            sorted_order = sorted_joint_idx[i]
            #print('sorted order', sorted_order, 'org verb', verbs[i])
            verbs_ordered = verbs[i][sorted_order]
            #print('ordered verbs ', verbs_ordered)
            sorted_beam_verb_ids.append(verbs_ordered)
            max_i = sorted_order[0]

            label_ids = beam_role_idx[i][6*max_i:6*(max_i+1)]
            #print('role place ', max_i, 6*max_i, 6*(max_i+1), label_ids)
            max_beam_role_label_ids.append(label_ids)
        #del f, g

        return sorted_beam_verb_ids, max_beam_role_label_ids


    def calculate_loss(self, verb_pred, gt_verbs, role_label_pred, gt_labels):

        batch_size = verb_pred.size()[0]
        criterion = nn.CrossEntropyLoss(ignore_index=self.vocab_size)
        loss = 0
        for i in range(batch_size):
            for index in range(gt_labels.size()[1]):
                frame_loss = 0
                verb_loss = utils.cross_entropy_loss(verb_pred[i], gt_verbs[i])
                #frame_loss = criterion(role_label_pred[i], gt_labels[i,index])
                for j in range(0, self.max_role_count):
                    frame_loss += utils.cross_entropy_loss(role_label_pred[i][j], gt_labels[i,index,j] ,self.vocab_size)
                frame_loss = verb_loss + frame_loss/len(self.encoder.verb2_role_dict[self.encoder.verb_list[gt_verbs[i]]])
                #print('frame loss', frame_loss)
                loss += (frame_loss + verb_loss)


        final_loss = loss/batch_size
        #print('loss :', final_loss)
        return final_loss

    def calculate_eval_loss(self, verb_pred, gt_verbs, role_label_pred, gt_labels):

        batch_size = verb_pred.size()[0]
        loss = 0
        sorted_idx = torch.sort(verb_pred, 1, True)[1]
        pred_verbs = sorted_idx[:,0]
        #print('eval pred verbs :', pred_verbs)
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
                print('frame loss', frame_loss, frame_loss.size())
                loss += frame_loss


        final_loss = loss/batch_size
        #print('loss :', final_loss)
        return final_loss
