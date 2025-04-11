from utils import *
from tqdm import tqdm
import torch.nn.functional as F
import logging


def base_train(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    ta = Averager()
    model = model.train()
    # standard classification for pretrain
    tqdm_gen = tqdm(trainloader)
    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cuda() for _ in batch]

        logits, _, _ = model(data)
        logits = logits[:, :args.base_class]
        loss = F.cross_entropy(logits, train_label)
        acc = count_acc(logits, train_label)

        total_loss = loss

        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
        tl.add(total_loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    tl = tl.item()
    ta = ta.item()
    return tl, ta

def replace_base_fc(trainset, transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            model.module.mode = 'encoder'
            embedding = model(data)

            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    model.module.fc_real.weight.data[:args.base_class] = proto_list

    corr_matrix = torch.zeros(proto_list.size(0), proto_list.size(0))
    for i in range(proto_list.size(0)):
        for j in range(proto_list.size(0)):
            corr_matrix[i][j] = model.module.RS(proto_list[i], proto_list[j])

    return model, corr_matrix


def inc_train_and_test(model, trainloader, testloader, optimizer, scheduler, epoch, args, session):
    # Training
    tl = Averager()

    for i, batch in enumerate(trainloader, 1):
        data, train_label = [_.cuda() for _ in batch]
        adjusted_labels = train_label - args.base_class - (session - 1) * args.way

        logits = model.module.get_inc_output(data, session)
        loss = F.cross_entropy(logits, adjusted_labels)
        total_loss = loss

        tl.add(total_loss.item())

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    if epoch % 10 == 0:

        model = model.eval()
        va = Averager()

        with torch.no_grad():
            for i, batch in enumerate(testloader, 1):
                data, test_label = [_.cuda() for _ in batch]
                adjusted_labels = test_label - args.base_class - (session - 1) * args.way

                logits = model.module.get_inc_output(data, session)
                acc = count_acc(logits, adjusted_labels)
                va.add(acc)

        test_acc = va.item()
        print(f"Testing Accuracy: {test_acc:.4f}")

    return tl.item()


def test(model, testloader, epoch, args, session, result_list=None, co_matrices=None):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    va5= Averager()
    lgt=torch.tensor([])
    lbs=torch.tensor([])
    with torch.no_grad():
        for i, batch in enumerate(testloader, 1):
            data, test_label = [_.cuda() for _ in batch]
            if co_matrices is None:

                logits, _, _ = model(data)
                logits = logits[:, :test_class]
                loss = F.cross_entropy(logits, test_label)
                acc = count_acc(logits, test_label)
                top5acc = count_acc_topk(logits, test_label)

                vl.add(loss.item())
                va.add(acc)
                va5.add(top5acc)

                lgt = torch.cat([lgt, logits.cpu()])
                lbs = torch.cat([lbs, test_label.cpu()])
                continue

            logits, feature, maps = model(data)
            logits = logits[:, :test_class]
            selected_sessions = decide_sessions(feature, args, co_matrices, model)

            total_acc = 0.0
            for idx, session in enumerate(selected_sessions):
                logits_sample = logits[idx].clone()
                predictions = F.softmax(logits_sample)
                K = logits_sample.shape[-1]  # assuming logits_sample is a 1D array with length = number of classes
                entropy = compute_entropy(predictions, K)

                if session == 0:
                    prototypes = model.module.fc.weight[:args.base_class]
                    similarities = F.linear(F.normalize(feature[idx].unsqueeze(0), p=2), F.normalize(prototypes, p=2))
                    logits_sample[:args.base_class] += similarities.squeeze() * entropy
                    global_pred = torch.argmax(logits_sample).item()
                else:

                    encoder_feature = model.module.attentions[session - 1](maps[idx].unsqueeze(0))
                    encoder_feature = F.adaptive_avg_pool2d(encoder_feature, 1)
                    encoder_feature = encoder_feature.squeeze(-1).squeeze(-1)
                    similarities = F.linear(F.normalize(encoder_feature, p=2, dim=1),
                                  F.normalize(model.module.fc_auxs[session - 1].weight, p=2, dim=1))

                    logits_sample[args.base_class + (session - 1) * args.way:args.base_class + session * args.way] \
                        += similarities.squeeze() * entropy

                    global_pred = torch.argmax(logits_sample).item()

                lgt = torch.cat([lgt, logits_sample.cpu()])
                lbs = torch.cat([lbs, test_label.cpu()])

                is_correct = (global_pred == test_label[idx].item())
                total_acc += is_correct

            avg_acc = total_acc / len(test_label)
            va.add(avg_acc)

        vl = vl.item()
        va = va.item()
        va5= va5.item()
        
        logging.info('epo {}, test, loss={:.4f} acc={:.4f}, acc@5={:.4f}'.format(epoch, vl, va, va5))

        lgt=lgt.view(-1, test_class)
        lbs=lbs.view(-1)
            
        if session > 0:
            cm = confmatrix(lgt,lbs)
            perclassacc = cm.diagonal()
            seenac = np.mean(perclassacc[:args.base_class])
            unseenac = np.mean(perclassacc[args.base_class:])
            
            result_list.append(f"Seen Acc:{seenac}  Unseen Acc:{unseenac}")
            return vl, (seenac, unseenac, va)
        else:
            return vl, va

def compute_entropy(predictions, K):
    log_preds = np.log(predictions)
    entropy = 1 + np.sum(predictions * log_preds, axis=-1) / np.log(K)
    return entropy

def decide_sessions(embeddings, args, co_matrices, model):
    # Ensure embeddings have the shape [batch_size, embedding_size]
    assert embeddings.dim() == 2

    batch_size = embeddings.size(0)
    all_selected_sessions = []

    for i in range(batch_size):
        embedding = embeddings[i]
        session_scores = []

        for session_idx in range(len(co_matrices)):
            if session_idx == 0:
                class_count = args.base_class
            else:
                class_count = args.way

            # Extract prototypes for the session
            if session_idx == 0:
                prototypes = model.module.fc.weight[:args.base_class]
            else:
                prototypes = model.module.fc.weight[args.base_class +
                                             (session_idx - 1) * args.way:args.base_class + session_idx * args.way]

            # Ensure prototypes have the shape [class_count, embedding_size]
            assert prototypes.shape == (class_count, embedding.size(-1))

            corr_vector = torch.zeros(prototypes.size(0)).cuda()
            for i in range(prototypes.size(0)):
                corr_vector[i] = model.module.RS(embedding, prototypes[i])

            _, session_score = best_matching_row(corr_vector, co_matrices[session_idx])
            session_scores.append(session_score)

        selected_session = torch.argmax(torch.tensor(session_scores))
        all_selected_sessions.append(selected_session.item())

    return all_selected_sessions

def best_matching_row(matrix, co_matrix):
    # Get the similarity for each row
    row_sims = torch.sum(matrix * co_matrix, dim=1)

    # Get the index of the row with the highest similarity
    _, best_row_idx = torch.max(row_sims, dim=0)

    return best_row_idx, row_sims[best_row_idx]
