__all__ = ["OffSerialTrainer"]

from cmath import inf
import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.tensorboard_setup import add_scalars
from utils.tensorboard_setup import tb_tags
from utils.common_utils import ModuleOnDevice


class OffSerialTrainer:
    def __init__(self, alg, sampler, buffer, evaluator, **kwargs):
        self.alg = alg
        self.sampler = sampler
        self.buffer = buffer
        self.per_flag = kwargs["buffer_name"] == "prioritized_replay_buffer"
        self.evaluator = evaluator

        # create center network
        self.networks = self.alg.networks
        self.sampler.networks = self.networks
        self.evaluator.networks = self.networks

        # initialize center network
        if kwargs["ini_network_dir"] is not None:
            self.networks.load_state_dict(torch.load(kwargs["ini_network_dir"]))

        self.replay_batch_size = kwargs["replay_batch_size"]
        self.max_iteration = kwargs["max_iteration"]
        self.sample_interval = kwargs.get("sample_interval", 1)
        self.log_save_interval = kwargs["log_save_interval"]
        self.apprfunc_save_interval = kwargs["apprfunc_save_interval"]
        self.eval_interval = kwargs["eval_interval"]
        self.best_tar = -inf
        self.save_folder = kwargs["save_folder"]
        self.iteration = 0

        self.writer = SummaryWriter(log_dir=self.save_folder, flush_secs=20)
        # flush tensorboard at the beginning
        add_scalars(
            {tb_tags["alg_time"]: 0, tb_tags["sampler_time"]: 0}, self.writer, 0
        )
        self.writer.flush()

        # pre sampling
        while self.buffer.size < kwargs["buffer_warm_size"]:
            samples, _ = self.sampler.sample()
            self.buffer.add_batch(samples)

        self.use_gpu = kwargs["use_gpu"]
        self.cuda_num=kwargs.get("device")
        self.device=torch.device(f"cuda:{self.cuda_num}")
        if self.use_gpu:
            #self.networks.cuda()
            print(self.device)
            self.networks.to(self.device)

        self.start_time = time.time()

    def step(self):
        # sampling
        #print(self.device)
        #self.networks.to(self.device)
        sampler_tb_dict = {}
        if self.iteration % self.sample_interval == 0:
            with ModuleOnDevice(self.networks, "cpu"):
                sampler_samples, sampler_tb_dict = self.sampler.sample()
            self.buffer.add_batch(sampler_samples)
        self.networks.to(self.device)
        # replay
        replay_samples = self.buffer.sample_batch(self.replay_batch_size)

        # learning
        if self.use_gpu:
            for k, v in replay_samples.items():
                #replay_samples[k] = v.cuda()
                replay_samples[k] = v.to(self.device)

        if self.per_flag:
            alg_tb_dict, idx, new_priority = self.alg.local_update(
                replay_samples, self.iteration
            )
            self.buffer.update_batch(idx, new_priority)
        else:
            alg_tb_dict = self.alg.local_update(replay_samples, self.iteration)

        # log
        if self.iteration % self.log_save_interval == 0:
            print("Iter = ", self.iteration)
            add_scalars(alg_tb_dict, self.writer, step=self.iteration)
            add_scalars(sampler_tb_dict, self.writer, step=self.iteration)

            # evaluate
            if self.iteration % self.eval_interval == 0:
                with ModuleOnDevice(self.networks, "cpu"):
                    total_avg_return, std = self.evaluator.run_evaluation_withstd(self.iteration)
                    print("avg_return = {} std = {}!".format(str(total_avg_return), str(std)))
                if (
                        total_avg_return >= self.best_tar
                        and self.iteration >= self.max_iteration / 100
                ):
                    self.best_tar = total_avg_return
                    print("Best return = {}!".format(str(self.best_tar)))

                    for filename in os.listdir(self.save_folder + "/apprfunc/"):
                        if filename.endswith("_opt.pkl"):
                            os.remove(self.save_folder + "/apprfunc/" + filename)

                    torch.save(
                        self.networks.state_dict(),
                        self.save_folder
                        + "/apprfunc/apprfunc_{}_opt.pkl".format(self.iteration),
                    )

                self.writer.add_scalar(
                    tb_tags["Buffer RAM of RL iteration"],
                    self.buffer.__get_RAM__(),
                    self.iteration,
                )
                self.writer.add_scalar(
                    tb_tags["TAR of RL iteration"], total_avg_return, self.iteration
                )
                self.writer.add_scalar(
                    tb_tags["TAR of replay samples"],
                    total_avg_return,
                    self.iteration * self.replay_batch_size,
                )
                self.writer.add_scalar(
                    tb_tags["TAR of total time"],
                    total_avg_return,
                    int(time.time() - self.start_time),
                )
            self.writer.add_scalar(
                tb_tags["TAR of collected samples"],
                total_avg_return,
                self.sampler.get_total_sample_number(),
            )

        # save
        if self.iteration % self.apprfunc_save_interval == 0:
            self.save_apprfunc()

    def train(self):
        print(self.networks)
        while self.iteration < self.max_iteration:
            self.step()
            self.iteration += 1

        self.save_apprfunc()
        self.writer.flush()

    def save_apprfunc(self):
        torch.save(
            self.networks.state_dict(),
            self.save_folder + "/apprfunc/apprfunc_{}.pkl".format(self.iteration),
        )


def create_trainer(alg, sampler, buffer, evaluator, **kwargs):
    trainer = OffSerialTrainer(alg, sampler,buffer, evaluator, **kwargs)
    print("Create trainer successfully!")
    return trainer




class OfflineSerialTrainer:
    def __init__(self, dataset,alg, sampler, buffer, evaluator, **kwargs):
        self.alg = alg
        self.buffer = buffer
        self.buffer.add_data_to_buffer(dataset)
        self.per_flag = kwargs["buffer_name"] == "prioritized_replay_buffer"
        self.evaluator = evaluator

        # create center network
        self.networks = self.alg.networks
        self.evaluator.networks = self.networks

        # initialize center network
        if kwargs["ini_network_dir"] is not None:
            self.networks.load_state_dict(torch.load(kwargs["ini_network_dir"]))

        self.replay_batch_size = kwargs["replay_batch_size"]
        self.max_iteration = kwargs["max_iteration"]
        self.log_save_interval = kwargs["log_save_interval"]
        self.apprfunc_save_interval = kwargs["apprfunc_save_interval"]
        self.eval_interval = kwargs["eval_interval"]
        self.best_tar = -inf
        self.save_folder = kwargs["save_folder"]
        self.iteration = 0

        self.writer = SummaryWriter(log_dir=self.save_folder, flush_secs=20)
        # flush tensorboard at the beginning
        add_scalars(
            {tb_tags["alg_time"]: 0, tb_tags["sampler_time"]: 0}, self.writer, 0
        )
        self.writer.flush()



        self.use_gpu = kwargs["use_gpu"]
        self.cuda_num=kwargs.get("device")
        self.device=torch.device(f"cuda:{self.cuda_num}")
        if self.use_gpu:
            #self.networks.cuda()
            print(self.device)
            self.networks.to(self.device)

        self.start_time = time.time()

    def step(self):
        # sampling
        self.networks.to(self.device)
        sampler_tb_dict = {}


        # replay
        start_time = time.time()
        replay_samples = self.buffer.sample_batch(self.replay_batch_size)

        replay_samples_time = time.time()
        if self.iteration % 5000 == 0:
            print(f"replay_samples: {replay_samples_time-start_time:.4f} seconds")
        #print(replay_samples)
        # learning
        if self.use_gpu:
            for k, v in replay_samples.items():
                #replay_samples[k] = v.cuda()
                replay_samples[k] = v.to(self.device)
        local_update_start_time = time.time()

        if self.per_flag:
            alg_tb_dict, idx, new_priority = self.alg.local_update(
                replay_samples, self.iteration
            )
            self.buffer.update_batch(idx, new_priority)
        else:
            alg_tb_dict = self.alg.local_update(replay_samples, self.iteration)
        
        local_update_end_time = time.time()
        if self.iteration%5000==0:
            print(f"local_update: {local_update_end_time - local_update_start_time:.4f} seconds")


        # log
        if self.iteration % self.log_save_interval == 0:
            print("Iter = ", self.iteration)
            add_scalars(alg_tb_dict, self.writer, step=self.iteration)
            add_scalars(sampler_tb_dict, self.writer, step=self.iteration)

        # evaluate
        if self.iteration % self.eval_interval == 0:

            evaluate_start_time = time.time()
            with ModuleOnDevice(self.networks, "cpu"):


                total_avg_return,std = self.evaluator.run_evaluation_withstd(self.iteration)
                print("Epoch:",self.iteration)
                print("avg_return = {} std = {}!".format(str(total_avg_return),str(std)))
            evaluate_end_time = time.time()
            if self.iteration % 5000 == 0:
                print(f"evaluate: {evaluate_end_time - evaluate_start_time:.4f} seconds")
            if (
                total_avg_return >= self.best_tar
                and self.iteration >= self.max_iteration / 100
            ):
                self.best_tar = total_avg_return
                print("Best return = {}!".format(str(self.best_tar)))

                for filename in os.listdir(self.save_folder + "/apprfunc/"):
                    if filename.endswith("_opt.pkl"):
                        os.remove(self.save_folder + "/apprfunc/" + filename)

                torch.save(
                    self.networks.state_dict(),
                    self.save_folder
                    + "/apprfunc/apprfunc_{}_opt.pkl".format(self.iteration),
                )

            self.writer.add_scalar(
                tb_tags["Buffer RAM of RL iteration"],
                self.buffer.__get_RAM__(),
                self.iteration,
            )
            self.writer.add_scalar(
                tb_tags["TAR of RL iteration"], total_avg_return, self.iteration
            )
            self.writer.add_scalar(
                tb_tags["TAR of replay samples"],
                total_avg_return,
                self.iteration * self.replay_batch_size,
            )
            self.writer.add_scalar(
                tb_tags["TAR of total time"],
                total_avg_return,
                int(time.time() - self.start_time),
            )

        # save
        if self.iteration % self.apprfunc_save_interval == 0:
            self.save_apprfunc()
        end_time = time.time()
        if self.iteration % 5000 == 0:
            print("step_time:",end_time-start_time)

    def train(self):
        print(self.networks)
        while self.iteration < self.max_iteration:
            self.step()
            self.iteration += 1

        self.save_apprfunc()
        self.writer.flush()

    def save_apprfunc(self):
        torch.save(
            self.networks.state_dict(),
            self.save_folder + "/apprfunc/apprfunc_{}.pkl".format(self.iteration),
        )




def create_offlinetrainer(dataset,alg, sampler, buffer, evaluator, **kwargs):
    trainer = OfflineSerialTrainer(dataset,alg, sampler,buffer, evaluator, **kwargs)
    print("Create offline-trainer successfully!")
    return trainer



class FinetuneSerialTrainer:
    def __init__(self,dataset, alg, sampler, buffer,online_buffer, evaluator, **kwargs):
        self.alg = alg
        self.sampler = sampler
        self.buffer = buffer
        self.online_buffer = online_buffer
        self.per_flag = kwargs["buffer_name"] == "prioritized_replay_buffer"
        self.evaluator = evaluator

        # create center network
        self.networks = self.alg.networks
        self.sampler.networks = self.networks
        self.evaluator.networks = self.networks

        # initialize center network
        if kwargs["ini_network_dir"] is not None:
            self.networks.load_state_dict(torch.load(kwargs["ini_network_dir"]))

        self.replay_batch_size = kwargs["replay_batch_size"]
        self.max_iteration = kwargs["max_iteration"]
        self.utd_iteration = kwargs["utd_iteration"]
        self.utd = kwargs["utd"]
        self.sample_interval = kwargs.get("sample_interval", 1)
        self.log_save_interval = kwargs["log_save_interval"]
        self.apprfunc_save_interval = kwargs["apprfunc_save_interval"]
        self.eval_interval = kwargs["eval_interval"]
        self.best_tar = -inf
        self.save_folder = kwargs["save_folder"]
        self.iteration = 0

        self.writer = SummaryWriter(log_dir=self.save_folder, flush_secs=20)
        # flush tensorboard at the beginning
        add_scalars(
            {tb_tags["alg_time"]: 0, tb_tags["sampler_time"]: 0}, self.writer, 0
        )
        self.writer.flush()

        # pre sampling
        self.buffer.add_random_samples_to_buffer(dataset,kwargs["buffer_warm_size"])

        self.use_gpu = kwargs["use_gpu"]
        self.cuda_num=kwargs.get("device")
        self.device=torch.device(f"cuda:{self.cuda_num}")
        if self.use_gpu:
            #self.networks.cuda()
            self.networks.to(self.device)

        self.start_time = time.time()

    def step(self,utd=1):
        # sampling
        #self.networks.to(self.device)
        sampler_tb_dict = {}
        if self.iteration % self.sample_interval == 0:
            with ModuleOnDevice(self.networks, "cpu"):
                sampler_samples, sampler_tb_dict = self.sampler.sample()
            self.buffer.add_batch(sampler_samples)
            self.online_buffer.add_batch(sampler_samples)
        

        for i in range(utd):
            replay_samples = self.buffer.sample_batch(self.replay_batch_size)
            online_replay_samples = self.online_buffer.sample_batch(self.replay_batch_size)
            self.networks.to(self.device)
            if self.use_gpu:
                for k, v in replay_samples.items():
                    # replay_samples[k] = v.cuda()
                    replay_samples[k] = v.to(self.device)
                for k, v in online_replay_samples.items():
                    # replay_samples[k] = v.cuda()
                    online_replay_samples[k] = v.to(self.device)
            if self.per_flag:
                self.alg.value_update(replay_samples)
                alg_tb_dict, idx, new_priority = self.alg.local_update(
                    online_replay_samples, self.iteration
                )
                self.buffer.update_batch(idx, new_priority)
            else:
                self.alg.value_update(replay_samples)
                alg_tb_dict = self.alg.local_update(online_replay_samples, self.iteration)



        # log
        if self.iteration % self.log_save_interval == 0:
            print("Iter = ", self.iteration)
            add_scalars(alg_tb_dict, self.writer, step=self.iteration)
            add_scalars(sampler_tb_dict, self.writer, step=self.iteration)

            # evaluate
            if self.iteration % self.eval_interval == 0:
                with ModuleOnDevice(self.networks, "cpu"):
                    total_avg_return, std = self.evaluator.run_evaluation_withstd(self.iteration)
                    print("avg_return = {} std = {}!".format(str(total_avg_return), str(std)))
                if (
                        total_avg_return >= self.best_tar
                        and self.iteration >= self.max_iteration / 100
                ):
                    self.best_tar = total_avg_return
                    print("Best return = {}!".format(str(self.best_tar)))

                    for filename in os.listdir(self.save_folder + "/apprfunc/"):
                        if filename.endswith("_opt.pkl"):
                            os.remove(self.save_folder + "/apprfunc/" + filename)

                    torch.save(
                        self.networks.state_dict(),
                        self.save_folder
                        + "/apprfunc/apprfunc_{}_opt.pkl".format(self.iteration),
                    )

                self.writer.add_scalar(
                    tb_tags["Buffer RAM of RL iteration"],
                    self.buffer.__get_RAM__(),
                    self.iteration,
                )
                self.writer.add_scalar(
                    tb_tags["TAR of RL iteration"], total_avg_return, self.iteration
                )
                self.writer.add_scalar(
                    tb_tags["TAR of replay samples"],
                    total_avg_return,
                    self.iteration * self.replay_batch_size,
                )
                self.writer.add_scalar(
                    tb_tags["TAR of total time"],
                    total_avg_return,
                    int(time.time() - self.start_time),
                )
            self.writer.add_scalar(
                tb_tags["TAR of collected samples"],
                total_avg_return,
                self.sampler.get_total_sample_number(),
            )

        # save
        if self.iteration % self.apprfunc_save_interval == 0:
            self.save_apprfunc()

    def pre_step(self, utd=1):
        # sampling
        # self.networks.to(self.device)
        sampler_tb_dict = {}
        if self.iteration % self.sample_interval == 0:
            with ModuleOnDevice(self.networks, "cpu"):
                sampler_samples, sampler_tb_dict = self.sampler.sample()
            self.buffer.add_batch(sampler_samples)
            self.online_buffer.add_batch(sampler_samples)

        for i in range(utd):
            replay_samples = self.buffer.sample_batch(self.replay_batch_size)

            self.networks.to(self.device)
            if self.use_gpu:
                for k, v in replay_samples.items():
                    # replay_samples[k] = v.cuda()
                    replay_samples[k] = v.to(self.device)

            if self.per_flag:
                alg_tb_dict, idx, new_priority = self.alg.local_update_wo_policy(
                    replay_samples, self.iteration
                )
                self.buffer.update_batch(idx, new_priority)
            else:
                alg_tb_dict = self.alg.local_update_wo_policy(replay_samples, self.iteration)

        # log
        if self.iteration % self.log_save_interval == 0:
            print("Iter = ", self.iteration)
            add_scalars(alg_tb_dict, self.writer, step=self.iteration)
            add_scalars(sampler_tb_dict, self.writer, step=self.iteration)

            # evaluate
            if self.iteration % self.eval_interval == 0:
                with ModuleOnDevice(self.networks, "cpu"):
                    total_avg_return, std = self.evaluator.run_evaluation_withstd(self.iteration)
                    print("avg_return = {} std = {}!".format(str(total_avg_return), str(std)))
                if (
                        total_avg_return >= self.best_tar
                        and self.iteration >= self.max_iteration / 100
                ):
                    self.best_tar = total_avg_return
                    print("Best return = {}!".format(str(self.best_tar)))

                    for filename in os.listdir(self.save_folder + "/apprfunc/"):
                        if filename.endswith("_opt.pkl"):
                            os.remove(self.save_folder + "/apprfunc/" + filename)

                    torch.save(
                        self.networks.state_dict(),
                        self.save_folder
                        + "/apprfunc/apprfunc_{}_opt.pkl".format(self.iteration),
                    )

                self.writer.add_scalar(
                    tb_tags["Buffer RAM of RL iteration"],
                    self.buffer.__get_RAM__(),
                    self.iteration,
                )
                self.writer.add_scalar(
                    tb_tags["TAR of RL iteration"], total_avg_return, self.iteration
                )
                self.writer.add_scalar(
                    tb_tags["TAR of replay samples"],
                    total_avg_return,
                    self.iteration * self.replay_batch_size,
                )
                self.writer.add_scalar(
                    tb_tags["TAR of total time"],
                    total_avg_return,
                    int(time.time() - self.start_time),
                )
            self.writer.add_scalar(
                tb_tags["TAR of collected samples"],
                total_avg_return,
                self.sampler.get_total_sample_number(),
            )

        # save
        if self.iteration % self.apprfunc_save_interval == 0:
            self.save_apprfunc()

    
    
    def pre_step_online(self, utd=1):
        # sampling
        # self.networks.to(self.device)
        sampler_tb_dict = {}
        if self.iteration % self.sample_interval == 0:
            with ModuleOnDevice(self.networks, "cpu"):
                sampler_samples, sampler_tb_dict = self.sampler.sample()
            self.buffer.add_batch(sampler_samples)
            self.online_buffer.add_batch(sampler_samples)

        for i in range(utd):
            replay_samples = self.online_buffer.sample_batch(self.replay_batch_size)

            self.networks.to(self.device)
            if self.use_gpu:
                for k, v in replay_samples.items():
                    # replay_samples[k] = v.cuda()
                    replay_samples[k] = v.to(self.device)

            if self.per_flag:
                alg_tb_dict, idx, new_priority = self.alg.local_update_wo_policy(
                    replay_samples, self.iteration
                )
                self.online_buffer.update_batch(idx, new_priority)
            else:
                alg_tb_dict = self.alg.local_update_wo_policy(replay_samples, self.iteration)

        # log
        if self.iteration % self.log_save_interval == 0:
            print("Iter = ", self.iteration)
            add_scalars(alg_tb_dict, self.writer, step=self.iteration)
            add_scalars(sampler_tb_dict, self.writer, step=self.iteration)

            # evaluate
            if self.iteration % self.eval_interval == 0:
                with ModuleOnDevice(self.networks, "cpu"):
                    total_avg_return, std = self.evaluator.run_evaluation_withstd(self.iteration)
                    print("avg_return = {} std = {}!".format(str(total_avg_return), str(std)))
                if (
                        total_avg_return >= self.best_tar
                        and self.iteration >= self.max_iteration / 100
                ):
                    self.best_tar = total_avg_return
                    print("Best return = {}!".format(str(self.best_tar)))

                    for filename in os.listdir(self.save_folder + "/apprfunc/"):
                        if filename.endswith("_opt.pkl"):
                            os.remove(self.save_folder + "/apprfunc/" + filename)

                    torch.save(
                        self.networks.state_dict(),
                        self.save_folder
                        + "/apprfunc/apprfunc_{}_opt.pkl".format(self.iteration),
                    )

                self.writer.add_scalar(
                    tb_tags["Buffer RAM of RL iteration"],
                    self.buffer.__get_RAM__(),
                    self.iteration,
                )
                self.writer.add_scalar(
                    tb_tags["TAR of RL iteration"], total_avg_return, self.iteration
                )
                self.writer.add_scalar(
                    tb_tags["TAR of replay samples"],
                    total_avg_return,
                    self.iteration * self.replay_batch_size,
                )
                self.writer.add_scalar(
                    tb_tags["TAR of total time"],
                    total_avg_return,
                    int(time.time() - self.start_time),
                )
            self.writer.add_scalar(
                tb_tags["TAR of collected samples"],
                total_avg_return,
                self.sampler.get_total_sample_number(),
            )

        # save
        if self.iteration % self.apprfunc_save_interval == 0:
            self.save_apprfunc()    

    
    
    def train(self):
        print(self.networks)
        print(self.utd_iteration)
        while self.iteration < self.max_iteration:
            if self.iteration < 1000:
                self.pre_step(utd=self.utd)
                self.iteration += 1
            elif self.iteration >= 1000 and self.iteration < self.utd_iteration:
                self.pre_step_online(utd=self.utd)
                self.iteration += 1
            else:
                self.step(utd=self.utd)
                self.iteration += 1
        
                


        self.save_apprfunc()
        self.writer.flush()

    def save_apprfunc(self):
        torch.save(
            self.networks.state_dict(),
            self.save_folder + "/apprfunc/apprfunc_{}.pkl".format(self.iteration),
        )


def create_finetunetrainer(dataset,alg, sampler, buffer,online_buffer, evaluator, **kwargs):
    trainer = FinetuneSerialTrainer(dataset,alg, sampler,buffer,online_buffer, evaluator, **kwargs)
    print("Create trainer successfully!")
    return trainer








