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
                    if label not in self.label_list:
                        if label not in label_frequency:
                            label_frequency[label] = 1
                        else:
                            label_frequency[label] += 1
                        #only labels occur at least 20 times are considered
                        if label_frequency[label] == 20:
                            self.label_list.append(label)

        print('train set stats: \n\t verb count:', len(self.verb_list), '\n\t role count:',len(self.role_list),
              '\n\t label count:', len(self.label_list) ,
              '\n\t max role count:', self.max_role_count)


        verb2role_list = []
        for verb_id in range(len(self.verb_list)):
            current_role_list = self.verb2_role_dict[self.verb_list[verb_id]]

            role_verb = []
            for role in current_role_list:
                role_id = self.role_list.index(role)
                role_verb.append(role_id)

            padding_count = self.max_role_count - len(current_role_list)

            for i in range(padding_count):
                role_verb.append(len(self.role_list))

            verb2role_list.append(torch.tensor(role_verb))

        self.verb2role_list = torch.stack(verb2role_list)
        '''print('verb to role list :', self.verb2role_list.size())

        print('unit test verb and roles: \n')
        verb_test = [4,57,367]
        for verb_id in verb_test:
            print('verb :', self.verb_list[verb_id])

            role_list = self.verb2role_list[verb_id]

            for role in role_list:
                if role != len(self.role_list):
                    print('role : ', self.role_list[role])'''

    def encode(self, item):
        verb = self.verb_list.index(item['verb'])
        roles = self.get_role_ids(verb)
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

    def get_role_count(self, verb_id):
        return len(self.verb2_role_dict[self.verb_list[verb_id]])

    def get_role_ids_batch(self, verbs):
        role_batch_list = []

        for verb_id in verbs:
            role_ids = self.get_role_ids(verb_id)
            role_batch_list.append(role_ids)

        return torch.tensor(role_batch_list)

    def get_role_ids(self, verb_id):

        return self.verb2role_list[verb_id]

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
