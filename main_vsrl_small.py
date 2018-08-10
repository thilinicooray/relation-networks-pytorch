import torch
from imsitu_encoder import imsitu_encoder
from imsitu_loader import imsitu_loader
from imsitu_scorer_updated import imsitu_scorer
import json
import model_vsrl_small
import os
import utils
#from torchviz import make_dot
#from graphviz import Digraph


def train(model, train_loader, dev_loader, traindev_loader, optimizer, scheduler, max_epoch, model_dir, encoder, gpu_mode, clip_norm, lr_max, eval_frequency=4000):
    model.train()
    train_loss = 0
    total_steps = 0
    print_freq = 400
    dev_score_list = []

    '''if model.gpu_mode >= 0 :
        ngpus = 2
        device_array = [i for i in range(0,ngpus)]

        pmodel = torch.nn.DataParallel(model, device_ids=device_array)
    else:
        pmodel = model'''
    pmodel = model
    scheduler.step()

    '''if scheduler.get_lr()[0] < lr_max:
        scheduler.step()'''

    top1 = imsitu_scorer(encoder, 1, 3)
    top5 = imsitu_scorer(encoder, 5, 3)

    '''print('init param data check :')
    for f in model.parameters():
        if f.requires_grad:
            print(f.data.size())'''


    for epoch in range(max_epoch):
        #print('current sample : ', i, img.size(), verb.size(), roles.size(), labels.size())
        #sizes batch_size*3*height*width, batch*504*1, batch*6*190*1, batch*3*6*lebale_count*1
        mx = len(train_loader)
        for i, (img, verb, roles,labels) in enumerate(train_loader):
            #print("epoch{}-{}/{} batches\r".format(epoch,i+1,mx)) ,
            total_steps += 1

            if gpu_mode >= 0:
                img = torch.autograd.Variable(img.cuda())
                roles = torch.autograd.Variable(roles.cuda())
                verb = torch.autograd.Variable(verb.cuda())
                labels = torch.autograd.Variable(labels.cuda())
            else:
                img = torch.autograd.Variable(img)
                verb = torch.autograd.Variable(verb)
                roles = torch.autograd.Variable(roles)
                labels = torch.autograd.Variable(labels)

            optimizer.zero_grad()

            verb_predict, role_predict = pmodel(img, verb, roles)

            '''g = make_dot(verb_predict, model.state_dict())
            g.view()'''

            loss = model.calculate_loss(verb_predict, verb, role_predict, labels)
            print('current loss = ', loss)

            loss.backward()

            #torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)


            '''for param in filter(lambda p: p.requires_grad,model.parameters()):
                print(param.grad.data.sum())'''

            #start debugger
            #import pdb; pdb.set_trace()


            optimizer.step()

            '''print('grad check :')
            for f in model.parameters():
                print('data is')
                print(f.data)
                print('grad is')
                print(f.grad)'''

            train_loss += loss.item()

            top1.add_point(verb_predict, verb, role_predict, labels)
            top5.add_point(verb_predict, verb, role_predict, labels)


            if total_steps % print_freq == 0:
                top1_a = top1.get_average_results()
                top5_a = top5.get_average_results()
                print ("{},{},{}, {} , {}, loss = {:.2f}, avg loss = {:.2f}"
                       .format(total_steps-1,epoch,i, utils.format_dict(top1_a, "{:.2f}", "1-"),
                               utils.format_dict(top5_a,"{:.2f}","5-"), loss.item(),
                               train_loss / ((total_steps-1)%eval_frequency) ))


            if total_steps % eval_frequency == 0:
                top1, top5, val_loss = eval(model, dev_loader, encoder, gpu_mode)
                model.train()

                top1_avg = top1.get_average_results()
                top5_avg = top5.get_average_results()

                avg_score = top1_avg["verb"] + top1_avg["value"] + top1_avg["value-all"] + top5_avg["verb"] + \
                            top5_avg["value"] + top5_avg["value-all"] + top5_avg["value*"] + top5_avg["value-all*"]
                avg_score /= 8

                print ('Dev {} average :{:.2f} {} {}'.format(total_steps-1, avg_score*100,
                                                             utils.format_dict(top1_avg,'{:.2f}', '1-'),
                                                             utils.format_dict(top5_avg, '{:.2f}', '5-')))
                #print('Dev loss :', val_loss)

                dev_score_list.append(avg_score)
                max_score = max(dev_score_list)

                if max_score == dev_score_list[-1]:
                    torch.save(model.state_dict(), model_dir + "/{0}_small_smalllr.model".format(max_score))
                    print ('New best model saved! {0}'.format(max_score))

                #eval on the trainset

                '''top1, top5, val_loss = eval(model, traindev_loader, encoder, gpu_mode)
                model.train()

                top1_avg = top1.get_average_results()
                top5_avg = top5.get_average_results()

                avg_score = top1_avg["verb"] + top1_avg["value"] + top1_avg["value-all"] + top5_avg["verb"] + \
                            top5_avg["value"] + top5_avg["value-all"] + top5_avg["value*"] + top5_avg["value-all*"]
                avg_score /= 8

                print ('TRAINDEV {} average :{:.2f} {} {}'.format(total_steps-1, avg_score*100,
                                                                  utils.format_dict(top1_avg,'{:.2f}', '1-'),
                                                                  utils.format_dict(top5_avg, '{:.2f}', '5-')))'''

                print('current train loss', train_loss)
                train_loss = 0
                top1 = imsitu_scorer(encoder, 1, 3)
                top5 = imsitu_scorer(encoder, 5, 3)

            del verb_predict, role_predict, loss, img, verb, roles, labels
            #break
        print('Epoch ', epoch, ' completed!')
        #break

def eval(model, dev_loader, encoder, gpu_mode):
    model.eval()
    val_loss = 0

    print ('evaluating model...')
    top1 = imsitu_scorer(encoder, 1, 3)
    top5 = imsitu_scorer(encoder, 5, 3)
    with torch.no_grad():
        mx = len(dev_loader)
        for i, (img, verb, roles,labels) in enumerate(dev_loader):
            #print("{}/{} batches\r".format(i+1,mx)) ,
            '''im_data = torch.squeeze(im_data,0)
            im_info = torch.squeeze(im_info,0)
            gt_boxes = torch.squeeze(gt_boxes,0)
            num_boxes = torch.squeeze(num_boxes,0)
            verb = torch.squeeze(verb,0)
            roles = torch.squeeze(roles,0)
            labels = torch.squeeze(labels,0)'''

            if gpu_mode >= 0:
                img = torch.autograd.Variable(img.cuda())
                roles = torch.autograd.Variable(roles.cuda())
                verb = torch.autograd.Variable(verb.cuda())
                labels = torch.autograd.Variable(labels.cuda())
            else:
                img = torch.autograd.Variable(img)
                verb = torch.autograd.Variable(verb)
                roles = torch.autograd.Variable(roles)
                labels = torch.autograd.Variable(labels)

            verb_predict, role_predict = model.forward_eval(img)
            '''loss = model.calculate_eval_loss(verb_predict, verb, role_predict, labels)
            val_loss += loss.item()'''
            top1.add_point_eval(verb_predict, verb, role_predict, labels)
            top5.add_point_eval(verb_predict, verb, role_predict, labels)

            del verb_predict, role_predict, img, verb, roles, labels
            #break

    #return top1, top5, val_loss/mx

    return top1, top5, 0

def main():

    import argparse
    parser = argparse.ArgumentParser(description="imsitu VSRL. Training, evaluation and prediction.")
    parser.add_argument("--gpuid", default=-1, help="put GPU id > -1 in GPU mode", type=int)
    parser.add_argument("--command", choices = ["train", "eval", "resume", 'predict'], required = True)
    parser.add_argument("--weights_file", help="the model to start from")

    args = parser.parse_args()

    batch_size = 640
    lr = 5e-6
    lr = 0.0001
    lr_max = 5e-4
    lr_gamma = 0.1
    lr_step = 10
    clip_norm = 50
    weight_decay = 1e-4
    n_epoch = 500
    n_worker = 3

    dataset_folder = 'imSitu'
    imgset_folder = 'resized_256'

    train_set = json.load(open(dataset_folder + "/train.json"))
    encoder = imsitu_encoder(train_set)

    model = model_vsrl_small.RelationNetworks(encoder, args.gpuid)

    train_set = imsitu_loader(imgset_folder, train_set, encoder, model.train_preprocess())

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=n_worker)

    dev_set = json.load(open(dataset_folder +"/dev.json"))
    dev_set = imsitu_loader(imgset_folder, dev_set, encoder, model.train_preprocess())
    dev_loader = torch.utils.data.DataLoader(dev_set, batch_size=32, shuffle=True, num_workers=n_worker)

    traindev_set = json.load(open(dataset_folder +"/dev.json"))
    traindev_set = imsitu_loader(imgset_folder, traindev_set, encoder, model.train_preprocess())
    traindev_loader = torch.utils.data.DataLoader(traindev_set, batch_size=8, shuffle=True, num_workers=n_worker)

    if args.command == "resume":
        print ("loading model weights...")
        model.load_state_dict(torch.load(args.weights_file))

    if args.gpuid >= 0:
        #print('GPU enabled')
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
    #gradient clipping, grad check

    print('Model training started!')
    train(model, train_loader, dev_loader, traindev_loader, optimizer, scheduler, n_epoch, 'trained_models', encoder, args.gpuid, clip_norm, lr_max)



if __name__ == "__main__":
    main()






