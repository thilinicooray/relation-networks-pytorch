#original code https://github.com/my89/imSitu/blob/master/imsitu.py

import torch

class imsitu_scorer():
    def __init__(self, encoder,topk, nref, write_to_file=False):
        self.score_cards = []
        self.topk = topk
        self.nref = nref
        self.encoder = encoder
        self.write_to_file = write_to_file
        if self.write_to_file:
            self.predicted_situation = []
            self.gt_situation = []

    def clear(self):
        self.score_cards = {}

    def add_point(self, verb_predict, gt_verbs, labels_predict, gt_labels):
        #encoded predictions should be batch x verbs x values #assumes the are the same order as the references
        #encoded reference should be batch x 1+ references*roles,values (sorted)

        batch_size = verb_predict.size()[0]
        for i in range(batch_size):
            verb_pred = verb_predict[i]
            gt_verb = gt_verbs[i]
            label_pred = labels_predict[i]
            gt_label = gt_labels[i]

            #print('check sizes:', verb_pred.size(), gt_verb.size(), label_pred.size(), gt_label.size())
            sorted_idx = torch.sort(verb_pred, 0, True)[1]

            gt_v = gt_verb
            role_set = self.encoder.get_role_ids(gt_v)
            #print('sorted idx:',self.topk, sorted_idx[:self.topk], gt_v)
            #print('groud truth verb id:', gt_v)


            new_card = {"verb":0.0, "value":0.0, "value*":0.0, "n_value":0.0, "value-all":0.0, "value-all*":0.0}


            score_card = new_card

            verb_found = (torch.sum(sorted_idx[0:self.topk] == gt_v) == 1)
            if verb_found: score_card["verb"] += 1

            gt_role_count = self.encoder.get_role_count(gt_v)
            gt_role_list = self.encoder.verb2_role_dict[self.encoder.verb_list[gt_v]]
            score_card["n_value"] += gt_role_count

            all_found = True
            pred_list = []
            for k in range(0, self.encoder.get_max_role_count()):
                role_id = role_set[k]
                if role_id == len(self.encoder.role_list):
                    continue
                current_role = self.encoder.role_list[role_id]
                if current_role not in gt_role_list:
                    continue

                label_id = torch.max(label_pred[k],0)[1]
                pred_list.append(label_id.item())
                found = False
                for r in range(0,self.nref):
                    gt_label_id = gt_label[r][k]
                    #print('ground truth label id = ', gt_label_id)
                    if label_id == gt_label_id:
                        found = True
                        break
                if not found: all_found = False
                #both verb and at least one val found
                if found and verb_found: score_card["value"] += 1
                #at least one val found
                if found: score_card["value*"] += 1
            '''if self.topk == 1:
                print('predicted labels :',pred_list)'''
            #both verb and all values found
            score_card["value*"] /= gt_role_count
            score_card["value"] /= gt_role_count
            if all_found and verb_found: score_card["value-all"] += 1
            #all values found
            if all_found: score_card["value-all*"] += 1

            self.score_cards.append(new_card)

    def add_point_eval(self, verb_predict, gt_verbs, labels_predict, gt_labels):
        #encoded predictions should be batch x verbs x values #assumes the are the same order as the references
        #encoded reference should be batch x 1+ references*roles,values (sorted)

        batch_size = verb_predict.size()[0]
        for i in range(batch_size):
            verb_pred = verb_predict[i]
            gt_verb = gt_verbs[i]
            label_pred = labels_predict[i]
            gt_label = gt_labels[i]

            #print('check sizes:', verb_pred.size(), gt_verb.size(), label_pred.size(), gt_label.size())
            sorted_idx = torch.sort(verb_pred, 0, True)[1]
            #print('top 1:', sorted_idx[0])
            role_set = self.encoder.get_role_ids(sorted_idx[0])
            gt_v = gt_verb
            gt_role_set = self.encoder.get_role_ids(gt_v)
            #print('sorted idx:',self.topk, sorted_idx[:self.topk], gt_v)
            #print('groud truth verb id:', gt_v)
            #print('role sets :', role_set, gt_role_set)


            new_card = {"verb":0.0, "value":0.0, "value*":0.0, "n_value":0.0, "value-all":0.0, "value-all*":0.0}


            score_card = new_card

            verb_found = (torch.sum(sorted_idx[0:self.topk] == gt_v) == 1)
            if verb_found: score_card["verb"] += 1

            gt_role_count = self.encoder.get_role_count(gt_v)
            gt_role_list = self.encoder.verb2_role_dict[self.encoder.verb_list[gt_v]]
            score_card["n_value"] += gt_role_count

            all_found = True
            pred_list = []
            for k in range(0, self.encoder.get_max_role_count()):
                role_id = role_set[k]
                if role_id == len(self.encoder.role_list):
                    continue
                current_role = self.encoder.role_list[role_id]
                if current_role not in gt_role_list:
                    continue
                g_idx = (gt_role_set == role_id).nonzero()
                label_id = torch.max(label_pred[k],0)[1]
                pred_list.append(label_id.item())
                found = False
                for r in range(0,self.nref):
                    gt_label_id = gt_label[r][g_idx]
                    #print('ground truth label id = ', gt_label_id)
                    if label_id == gt_label_id:
                        found = True
                        break
                if not found: all_found = False

                #both verb and at least one val found
                if found and verb_found: score_card["value"] += 1
                #at least one val found
                if found: score_card["value*"] += 1
            '''if self.topk == 1:
                print('predicted labels :',pred_list)'''

            if len(pred_list) < gt_role_count:
                all_found = False
            #both verb and all values found
            score_card["value*"] /= gt_role_count
            score_card["value"] /= gt_role_count
            '''if all_found:
                print('all found role sets :pred, gt', role_set, gt_role_set)'''
            if all_found and verb_found: score_card["value-all"] += 1
            #all values found
            if all_found: score_card["value-all*"] += 1

            self.score_cards.append(new_card)

    def add_point_eval_new(self, verb_predict, gt_verbs, labels_predict, gt_labels):
        #encoded predictions should be batch x verbs x values #assumes the are the same order as the references
        #encoded reference should be batch x 1+ references*roles,values (sorted)

        batch_size = verb_predict.size()[0]
        for i in range(batch_size):
            verb_pred = verb_predict[i]
            gt_verb = gt_verbs[i]
            label_pred = labels_predict[i]
            gt_label = gt_labels[i]

            #print('check sizes:', verb_pred.size(), gt_verb.size(), label_pred.size(), gt_label.size())
            sorted_idx = torch.sort(verb_pred, 0, True)[1]
            #print('top 1:', sorted_idx[0])
            role_set = self.encoder.get_role_ids(sorted_idx[0])
            gt_v = gt_verb
            gt_role_set = self.encoder.get_role_ids(gt_v)
            #print('sorted idx:',self.topk, sorted_idx[:self.topk], gt_v)
            #print('groud truth verb id:', gt_v)
            #print('role sets :', role_set, gt_role_set)


            new_card = {"verb":0.0, "value":0.0, "value*":0.0, "n_value":0.0, "value-all":0.0, "value-all*":0.0}
            if self.write_to_file:
                gt_sit = [gt_v.item()]
                pred_sit = [sorted_idx[0:self.topk].item()]


            score_card = new_card

            verb_found = (torch.sum(sorted_idx[0:self.topk] == gt_v) == 1)
            if verb_found: score_card["verb"] += 1

            gt_role_count = self.encoder.get_role_count(gt_v)
            gt_role_list = self.encoder.verb2_role_dict[self.encoder.verb_list[gt_v]]
            score_card["n_value"] += gt_role_count

            all_found = True
            pred_list = []
            for k in range(0, gt_role_count):

                label_id = torch.max(label_pred[k],0)[1]
                pred_list.append(label_id.item())
                found = False
                for r in range(0,self.nref):
                    gt_label_id = gt_label[r][k]
                    if self.write_to_file:
                        pred_sit.append(label_id.item())
                        gt_sit.append(gt_label_id.item())
                    #print('ground truth label id = ', gt_label_id)
                    if label_id == gt_label_id:
                        found = True
                        break
                if not found: all_found = False

                #both verb and at least one val found
                if found and verb_found: score_card["value"] += 1
                #at least one val found
                if found: score_card["value*"] += 1
            '''if self.topk == 1:
                print('predicted labels :',pred_list)'''

            '''if len(pred_list) < gt_role_count:
                all_found = False'''
            #both verb and all values found
            score_card["value*"] /= gt_role_count
            score_card["value"] /= gt_role_count
            '''if all_found:
                print('all found role sets :pred, gt', role_set, gt_role_set)'''
            if all_found and verb_found: score_card["value-all"] += 1
            #all values found
            if all_found: score_card["value-all*"] += 1

            self.score_cards.append(new_card)
            if self.write_to_file:
                self.predicted_situation.append(pred_sit)
                self.gt_situation.append(gt_sit)


    def combine(self, rv, card):
        for (k,v) in card.items(): rv[k] += v

    def get_average_results(self, groups = []):
        #average across score cards.
        rv = {"verb":0, "value":0 , "value*":0 , "value-all":0, "value-all*":0}
        total_len = len(self.score_cards)
        remove_num = 3000
        for card in self.score_cards:
            print('verb val :', card["verb"])
            if card["verb"] < 1 and remove_num > 0:
                remove_num =- 1
                print('came here ', remove_num)
                continue
            rv["verb"] += card["verb"]
            rv["value-all"] += card["value-all"]
            #rv["value-all*"] += card["value-all*"]
            rv["value"] += card["value"]
            #rv["value*"] += card["value*"]
        total_len = total_len - remove_num
        rv["verb"] /= total_len
        rv["value-all"] /= total_len
        #rv["value-all*"] /= total_len
        rv["value"] /= total_len
        #rv["value*"] /= total_len

        return rv

    def rearrange_label_pred(self, pred):
        label_pred_per_role = []
        start_idx = self.encoder.role_start_idx
        end_idx = self.encoder.role_end_idx
        for j in range(0, self.encoder.get_max_role_count()):
            label_pred_per_role.append(pred[start_idx[j]:end_idx[j]])

        return label_pred_per_role

