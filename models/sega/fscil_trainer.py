import logging
from copy import deepcopy
import psutil
import torch.nn as nn
from base import Trainer
from dataloader.data_utils import get_dataloader
from utils import *
import torch.cuda as cuda

from .helper import *
from .Network import MYNET


class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_up_model()

        # Log model statistics only once at initialization
        self._log_model_stats()

    def set_up_model(self):
        self.model = MYNET(self.args, mode=self.args.base_mode)
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()

        if self.args.model_dir is not None:
            logging.info('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir,
                                              map_location={'cuda:3': 'cuda:0'})['params']
        else:
            logging.info('random init params')
            if self.args.start_session > 0:
                logging.info('WARING: Random init weights for new sessions!')
            self.best_model_dict = deepcopy(self.model.state_dict())

    def _log_model_stats(self):
        """Log model parameters and FLOPs"""
        try:
            # Get input size based on dataset
            if self.args.dataset in ['cifar100']:
                input_size = (1, 3, 32, 32)
            else:
                input_size = (1, 3, 224, 224)

            # Total model stats
            total_params = self.model.module.count_params()
            total_flops = self.model.module.calculate_flops(input_size)

            logging.info(f"Total Model Parameters: {total_params:,}")
            logging.info(f"Total Model FLOPs: {total_flops:,} ({total_flops / 1e9:.2f} GFLOPs)")

            # Per-session stats
            for session in range(self.args.sessions):
                session_params = self.model.module.count_params(session)
                session_flops = self.model.module.calculate_flops(input_size, session)
                logging.info(f"Session {session} Parameters: {session_params:,}")
                logging.info(f"Session {session} FLOPs: {session_flops:,} ({session_flops / 1e9:.2f} GFLOPs)")
        except Exception as e:
            logging.warning(f"Failed to log model stats: {str(e)}")
            logging.warning("Continuing with training...")

    def _log_memory_stats(self):
        """Log current memory usage"""
        # CPU Memory
        cpu_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        # GPU Memory
        gpu_memory_allocated = cuda.memory_allocated() / 1024 / 1024  # MB
        gpu_memory_cached = cuda.memory_reserved() / 1024 / 1024  # MB

        logging.info(f"CPU Memory: {cpu_memory:.2f} MB")
        logging.info(f"GPU Memory Allocated: {gpu_memory_allocated:.2f} MB")
        logging.info(f"GPU Memory Cached: {gpu_memory_cached:.2f} MB")

    def train(self):
        args = self.args
        t_start_time = time.time()

        # Init train statistics
        result_list = [args]
        self.co_matrix = []

        for session in range(args.start_session, args.sessions):
            train_set, trainloader, testloader = get_dataloader(args, session)
            self.model.load_state_dict(self.best_model_dict)

            session_start_time = time.time()
            # Log memory once at start of session
            logging.info(f"Memory stats for session {session}:")
            self._log_memory_stats()

            if session == 0:  # Base training
                if not args.only_do_incre:
                    logging.info(f'New classes for this session: {np.unique(train_set.targets)}')
                    optimizer, scheduler = get_optimizer(args, self.model)

                    for epoch in range(args.epochs_base):
                        start_time = time.time()

                        tl, ta = base_train(self.model, trainloader, optimizer, scheduler, epoch, args)
                        tsl, tsa = test(self.model, testloader, epoch, args, session, result_list=result_list)

                        # Save better model
                        if (tsa * 100) >= self.trlog['max_acc'][session]:
                            self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                            self.trlog['max_acc_epoch'] = epoch
                            save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                            torch.save(dict(params=self.model.state_dict()), save_model_dir)
                            self.best_model_dict = deepcopy(self.model.state_dict())
                            logging.info('********A better model is found!!**********')
                            logging.info('Saving model to :%s' % save_model_dir)

                        self.trlog['train_loss'].append(tl)
                        self.trlog['train_acc'].append(ta)
                        self.trlog['test_loss'].append(tsl)
                        self.trlog['test_acc'].append(tsa)
                        lrc = scheduler.get_last_lr()[0]

                        logging.info(
                            'epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                                epoch, lrc, tl, ta, tsl, tsa))
                        scheduler.step()

                    session_time = time.time() - session_start_time
                    logging.info(f'Session {session} completed in {session_time:.2f} seconds')

                    # Finish base train
                    logging.info('>>> Finish Base Train <<<')
                    result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                        session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session]))
                else:
                    logging.info('>>> Load Model &&& Finish base train...')
                    assert args.model_dir is not None

                if not args.not_data_init:
                    self.model.load_state_dict(self.best_model_dict)
                    self.model, covariance_matrix = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)
                    self.co_matrix.append(covariance_matrix.cuda())

                    best_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                    logging.info('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
                    self.best_model_dict = deepcopy(self.model.state_dict())
                    torch.save(dict(params=self.model.state_dict()), best_model_dir)

                    self.model.module.mode = 'avg_cos'
                    tsl, tsa = test(self.model, testloader, 0, args, session, result_list=result_list)
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        logging.info('The new best test acc of base session={:.3f}'.format(
                            self.trlog['max_acc'][session]))

            else:  # Incremental learning sessions
                logging.info(f"Training session: [{session}]")

                # Original incremental session code
                for param in self.model.parameters():
                    param.requires_grad = False
                for name, param in self.model.named_parameters():
                    if f'attentions.{session - 1}' in name or f'fc_auxs.{session - 1}' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

                trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
                total_epoch = 30
                milestone = [10, 20]
                optimizer, scheduler = get_optimizer_inc(trainable_params, args, milestone=milestone)

                self.model.module.mode = self.args.new_mode
                self.model.eval()
                trainloader.dataset.transform = testloader.dataset.transform

                covariance_matrix = self.model.module.update_fc(trainloader, np.unique(train_set.targets))
                self.co_matrix.append(covariance_matrix.cuda())

                for epoch in range(total_epoch):
                    tl = inc_train_and_test(self.model, trainloader, testloader, optimizer, scheduler, epoch, args,
                                            session)

                tsl, (seenac, unseenac, avgac) = test(self.model, testloader, 0, args, session, result_list=result_list)

                self.trlog['seen_acc'].append(float('%.3f' % (seenac * 100)))
                self.trlog['unseen_acc'].append(float('%.3f' % (unseenac * 100)))
                self.trlog['max_acc'][session] = float('%.3f' % (avgac * 100))
                self.best_model_dict = deepcopy(self.model.state_dict())

                session_time = time.time() - session_start_time
                logging.info(f'Session {session} completed in {session_time:.2f} seconds')
                logging.info(f"Session {session} ==> Seen Acc:{self.trlog['seen_acc'][-1]} "
                             f"Unseen Acc:{self.trlog['unseen_acc'][-1]} Avg Acc:{self.trlog['max_acc'][session]}")
                result_list.append('Session {}, test Acc {:.3f}\n'.format(session, self.trlog['max_acc'][session]))

        # Final timing
        total_time = (time.time() - t_start_time) / 60
        logging.info(f"Base Session Best epoch:{self.trlog['max_acc_epoch']}")
        logging.info('Total training time: {:.2f} minutes'.format(total_time))
        logging.info(self.args.time_str)

        # Save final results
        result_list, hmeans = postprocess_results(result_list, self.trlog)
        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)
        if not self.args.debug:
            save_result(args, self.trlog, hmeans)
