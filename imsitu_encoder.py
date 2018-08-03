import torch
import random
from collections import OrderedDict
import csv

#This is the class which encodes training set json in the following structure
#todo: the structure

class imsitu_encoder():
    def __init__(self, train_set):
        # json structure -> {<img_id>:{frames:[{<role1>:<label1>, ...},{}...], verb:<verb1>}}
        print('imsitu encoder initialization started.')
        self.verb_list = []
        self.role_list = []
        self.max_label_count = 3
        self.verb2_role_dict = {}
        self.label_list = ['#UNK#']
        label_frequency = {}
        self.max_role_count = 0
        self.role2_label = {}
        self.role2_verb = {}

        for img_id in train_set:
            img = train_set[img_id]
            current_verb = img['verb']
            if current_verb not in self.verb_list:
                self.verb_list.append(current_verb)
                self.verb2_role_dict[current_verb] = []

            for frame in img['frames']:
                for role,label in frame.items():
                    if role not in self.role_list:
                        self.role_list.append(role)
                    if role not in self.verb2_role_dict[current_verb]:
                        self.verb2_role_dict[current_verb].append(role)
                    if len(self.verb2_role_dict[current_verb]) > self.max_role_count:
                        self.max_role_count = len(self.verb2_role_dict[current_verb])
                    if role not in self.role2_label:
                        self.role2_label[role] = []
                    if label not in self.role2_label[role]:
                        self.role2_label[role].append(label)
                    if role not in self.role2_verb:
                        self.role2_verb[role] = []
                    if current_verb not in self.role2_verb[role]:
                        self.role2_verb[role].append(current_verb)
                    if label not in self.label_list:
                        if label not in label_frequency:
                            label_frequency[label] = 1
                        else:
                            label_frequency[label] += 1
                        #only labels occur at least 5 times are considered
                        if label_frequency[label] == 20:
                            self.label_list.append(label)

        print('train set stats: \n\t verb count:', len(self.verb_list), '\n\t role count:',len(self.role_list),
              '\n\t label count:', len(self.label_list) ,
              '\n\t max role count:', self.max_role_count)

        #self.save_stat(self.role2_label, 'role2label')
        #self.save_stat(self.role2_verb, 'role2verb')

        self.verb_embedding = torch.squeeze(torch.eye(len(self.verb_list)).view(len(self.verb_list),len(self.verb_list),1))
        self.role_embedding = torch.squeeze(torch.eye(len(self.role_list)).view(len(self.role_list),len(self.role_list),1))
        self.label_embedding = torch.squeeze(torch.eye(len(self.label_list)).view(len(self.label_list),len(self.label_list),1))

        '''print('embedding sizes for verb, role and label ', self.verb_embedding.size(), self.role_embedding.size(),
              self.label_embedding.size() )'''

        verb2role_embedding_list = []
        for verb_id in range(len(self.verb_list)):
            current_role_list = self.verb2_role_dict[self.verb_list[verb_id]]

            role_embedding_verb = []
            for role in current_role_list:
                role_id = self.role_list.index(role)
                embedding = self.role_embedding[role_id]
                role_embedding_verb.append(embedding)

            padding_count = self.max_role_count - len(current_role_list)

            for i in range(padding_count):
                role_embedding_verb.append(torch.zeros(len(self.role_list)))

            verb2role_embedding_list.append(torch.stack(role_embedding_verb))

        self.verb2role_embedding = torch.stack(verb2role_embedding_list)

        '''print('Final check')
        rand_verb = random.choice(self.verb_list)
        rand_verb_id = self.verb_list.index(rand_verb)
        print('selected verb and id :', rand_verb, rand_verb_id)
        print('verb embedding \n', self.verb_embedding[rand_verb_id].size())
        print(self.verb_embedding[rand_verb_id].t())
        print('role details : \n')
        for role in verb2_role_dict[rand_verb]:
            print(role, self.role_list.index(role))
            idx = verb2_role_dict[rand_verb].index(role)
            print('role embedding \n', self.verb2role_embedding[rand_verb_id][idx].t())'''


    '''def save_stat(self, dict, filename):
        newdict = {}
        for y in dict.items():
            newdict[y[0]] = len(y[1])

        d_sorted_by_value = OrderedDict(sorted(newdict.items(), key=lambda x: x[1], reverse=True))

        import pandas as pd

        (pd.DataFrame.from_dict(data=d_sorted_by_value, orient='index')
         .to_csv(filename+'.csv', header=False))'''


    def encode(self, item):
        verb = self.verb_list.index(item['verb'])
        roles = self.get_role_ids(item['verb'])
        labels = self.get_label_ids(item['frames'])

        #print('item encoding size : v r l', verb.size(), roles.size(), labels.size())
        #assuming labels are also in order of roles in encoder
        return verb, roles, labels

    def save_encoder(self):
        return None

    def load_encoder(self):
        return None

    def get_max_role_count(self):
        return self.max_role_count

    def get_num_verbs(self):
        return len(self.verb_list)

    def get_num_roles(self):
        return len(self.role_list)

    def get_num_labels(self):
        return len(self.label_list)

    def get_verb_encoding(self, verb_id):
        verbs = []

        for id in verb_id:
            verb = self.verb_embedding[id]
            verbs.append(verb)

        return torch.stack(verbs)

    def get_role_encoding(self, verb_id):
        verb_roles = []

        for id in verb_id:
            verb_r = self.verb2role_embedding[id]
            verb_roles.append(verb_r)

        return torch.stack(verb_roles)

    def get_role_count(self, verb_id):
        return len(self.verb2_role_dict[self.verb_list[verb_id]])

    def get_adj_matrix(self, verb_encoding):
        verb_ids = verb_encoding
        adj_matrix_list = []

        for id in verb_ids:
            adj = torch.zeros([self.max_role_count, self.max_role_count])
            actual_verb_count = self.get_role_count(id)
            adj[:actual_verb_count, : actual_verb_count] = 1
            adj_matrix_list.append(adj)

        return torch.stack(adj_matrix_list)

    def apply_mask(self, roles, val):
        mask = val.clone().fill_(1)
        batch_size = roles.size(0)
        for i in range(batch_size):
            for j in range(self.get_max_role_count()):
                role = roles[i][j]
                if role == 190:
                    #embedding[i][j] = embedding[i][j].fill_(0)
                    mask[i,j:] = 0
                    break

        return mask

    def get_role_ids(self, verb):
        #print('verb', verb)
        roles = self.verb2role_embedding[self.verb_list.index(verb)]
        #print('roles', roles)
        role_id = []

        for role in roles:
            val = role[role > 0]
            if len(val) > 0:
                _, id = torch.max(torch.unsqueeze(role,1), 0)
                role_id.append(id)

        pad = self.max_role_count - len(role_id)
        for j in range(pad):
            role_id.append(self.get_num_roles())

        return torch.tensor(role_id)

    def get_label_ids(self, frames):
        all_frame_id_list = []
        for frame in frames:
            label_id_list = []
            for role,label in frame.items():
                #use UNK when unseen labels come
                if label in self.label_list:
                    label_id = self.label_list.index(label)
                else:
                    label_id = self.label_list.index('#UNK#')

                label_id_list.append(label_id)

            role_padding_count = self.max_role_count - len(label_id_list)

            for i in range(role_padding_count):
                label_id_list.append(self.get_num_labels())

            all_frame_id_list.append(torch.tensor(label_id_list))

        labels = torch.stack(all_frame_id_list,0)

        return labels
