from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import numpy as np


class Scenary_Trainer():
    def __init__(self, model, criterion, watcher = None):
        self.model = model
        #self.optimizer = optimizer
        self.criterion = criterion
        #self.scheduler = scheduler    

        self.scenary = self.gen_scenary(model.layers)
        self.pass_scenary = self.gen_pass_scenary(model.layers)
        self.watcher = watcher
        self.train_loss_history = []
        self.test_loss_history = []
        

    def train(self, clear_data, noisy_data, clear_test, noisy_test ,  epochs, device = 'cpu'):
        self.model.to(device)
        previous_loss_value = 0
        degrade_counter = 0

        for epoch in range(epochs):
            self.model.train()
            current_mse = torch.tensor(0.0, dtype= torch.float64)
            test_mse = torch.tensor(0.0, dtype= torch.float64)
            
            self.optimizer.zero_grad()

            for cur_data_idx in range(clear_data.shape[0]):
                x = clear_data[cur_data_idx,:].type(torch.complex64).detach()
                y = noisy_data[cur_data_idx,:].type(torch.complex64).detach()

                x_hat = self.model(y)
                current_mse += self.criterion(x,x_hat)
            
            current_mse.backward(retain_graph=True)
            self.optimizer.step()
            self.train_loss_history.append(current_mse.detach().item())

            # -------------TESTING------------- 
            self.model.eval()
            with torch.no_grad():
                for cur_data_idx in range(clear_test.shape[0]):
                    
                    if epoch%5==0:
                        if cur_data_idx == 0:
                            self.model.is_save_log = True

                    x = clear_test[cur_data_idx,:].type(torch.complex64).detach()
                    y = noisy_test[cur_data_idx,:].type(torch.complex64).detach()

                    test_mse += self.criterion(x, self.model(y))
                    self.model.is_save_log = False
                    
                    #if epoch%5==0:
                        #if cur_data_idx == 0:
                                
                            #self.model.send_log_wandb(x,y)

            self.test_loss_history.append(test_mse.item())
            if epoch%5 == 0:
                print('Epoch: {0}. Train Loss : {1:9.5f}. Test Loss: {2:9.5f}'.format(epoch,current_mse.item(), test_mse.item()))

            if self.scheduler.optimizer.param_groups[0]['lr'] < 1e-10:
                break
            
            if test_mse.item() > previous_loss_value:
                degrade_counter += 1

                #if degrade_counter > 50:
                #    break
            
            elif degrade_counter > 0:
                degrade_counter -=1

            previous_loss_value = current_mse.item()
            
            self.scheduler.step(test_mse.detach().item())
            


    def train_with_scenary(self, clear_data, noisy_data, clear_test, noisy_test, epochs):
        for i in range(0, len(self.scenary)):
            print('\n', '======================================')
            print('\n', 'Stage ', i)

            if i == 0:
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 1.0e-2)
            else:
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 1.0e-3)

            self.scheduler = ReduceLROnPlateau(
                                        optimizer = self.optimizer,
                                        mode = 'min',
                                        factor = 0.05,
                                        patience = 10,
                                        threshold = 1e-3,
                                        threshold_mode = 'rel',
                                        cooldown = 10,
                                        min_lr = 1e-12 
            )

            if i%2==1:
                n = int((i+1)/2)
                self.model.set_params_prev_layer(n)
            
            self.model.setState(self.scenary[i], self.pass_scenary[i])
            self.train(clear_data, noisy_data, clear_test, noisy_test ,  epochs)

        print("===========================================================")
        print("===========================================================")
        print("DONE")



    def bin_to_dec(self, bin_str):
            if type(bin_str == str):
                return  int(bin_str , 2) 
            return -1


    def dec_to_bin(self, dec_val):
        return bin(dec_val)[2:]


    def gen_scenary(self, n_layers):
        n_zeros = n_layers - 1
        start_value = '1'
        end_value = '1'

        start_value += '0' * n_zeros
        end_value += '1' * n_zeros

        scenary = []

        current_dec = self.bin_to_dec(start_value)
        scenary.append(start_value)

        first_dec = self.bin_to_dec(start_value)
        second_dec = first_dec

        for i in range(1, n_layers):
            first = i*'0' + self.dec_to_bin(int(first_dec/2))
            scenary.append(first)
            first_dec = self.bin_to_dec(first)

            second_dec = second_dec + first_dec
            second = self.dec_to_bin(second_dec)
            scenary.append(second)

        return scenary


    def gen_pass_scenary(self, n_layers):
        start = '0' + '1'*(n_layers-1)
        pass_scen = []

        pass_scen.append(start)
        start_dec = self.bin_to_dec(start)

        current = start_dec
        for i in range(1, n_layers-1):
            
            start = (i+1)*'0' + self.dec_to_bin(start_dec >> 1)

            pass_scen.append(start)
            pass_scen.append(start)

            start_dec = self.bin_to_dec(start)

        end = '0' * n_layers
        pass_scen.append(end)
        pass_scen.append(end)
        return pass_scen




class Trainer():
    def __init__(self, net, cfg, optimizer, criterion):
        self.net = net
        self.cfg = cfg
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = cfg.epochs

        self.scenary = self.gen_scenary(cfg.layers)
        self.pass_scenary = self.gen_pass_scenary(cfg.layers)

        self.loss_history = []
    

    def train_with_scenary(self, u, h_data_noisy, data_noise_power, scen0):
        for i in range(0, len(self.scenary)):
            print('\n', 'Stage ', i)
            self.net.setState(self.scenary[i], self.pass_scenary[i])
            self.train(tr_round = i, u = u, h_data_noisy= h_data_noisy, data_noise_power = data_noise_power, scen0= scen0, epochs = self.epochs)


    def train_with_scenary_batch(self, u, h_data_noisy, data_noise_power, scen0):
        for i in range(0, len(self.scenary)):
            print('\n', 'Stage ', i)
            self.net.setState(self.scenary[i], self.pass_scenary[i])
            self.batch_train(self, u_batch= u, h_data_batch= h_data_noisy, noise_power_batch = data_noise_power, scen0= scen0, epochs = self.epochs)


    def allow_all(self):
        scenary = self.gen_scenary(self.cfg.layers)[-1]
        pass_scenary = self.gen_pass_scenary(self.cfg.layers)[-1]
        self.net.setState(scenary, pass_scenary)
        

    def batch_train(self, u_batch , h_data_batch, noise_power_batch, scen0, epochs):
        loss_history = []
        degrade_counter = 0
        flat_counter = 0
        prev_loss = 0 

        for i in range(epochs):
            batch_loss = torch.tensor([0.0], requires_grad = True)

            for sample in range(u_batch.shape[0]):
                h_rec = self.net.forward(u_batch[sample,:,:,:])
                loss_value = self.criterion(h_rec, h_data_batch[sample,:,:,:,:], noise_power_batch[sample], scen0)
                batch_loss = batch_loss + loss_value
            
            self.optimizer.zero_grad()
            batch_loss.backward(retain_graph = True)
            self.optimizer.step()

            if prev_loss < batch_loss: degrade_counter+=1 
            elif torch.isclose(prev_loss, batch_loss, atol=1e-06): flat_counter+=1
            else: 
                degrade_counter-=1
                flat_counter-=1

            if degrade_counter > 3:
                break
            
            if flat_counter > 4:
                break
            
            prev_loss = loss_value
            

            if i%1 ==0: print('epoch = ',i,', loss = ',np.round(batch_loss.item(), 6))
            loss_history.append(batch_loss.item())

        return loss_history



    def train(self, u , tr_round, h_data_noisy, data_noise_power, scen0, epochs):
        degrade_counter = 0
        flat_counter = 0

        prev_loss = 0
        for i in range(epochs):
            self.optimizer.zero_grad()
            h_rec = self.net.forward(u)
            loss_value = self.criterion(h_rec, h_data_noisy, data_noise_power, scen0) 

            if i%5 ==0: 
                self.loss_history.append(loss_value)
                print('round = ',tr_round,', loss = ',np.round(loss_value.item(), 6))

            loss_value.backward(retain_graph=True)
            self.optimizer.step()

            if prev_loss < loss_value: degrade_counter+=1 
            elif torch.isclose(prev_loss, loss_value, atol=1e-06): flat_counter+=1
            else: 
                degrade_counter-=1
                flat_counter-=1

            #for p in lamp.parameters():
            #    p.data.clamp_(min = 0.00001, max = 500)
            
            if degrade_counter > 3:
                break
            
            if flat_counter > 50:
                break
            
            prev_loss = loss_value
            
        
    def bin_to_dec(self, bin_str):
        if type(bin_str == str):
            return  int(bin_str , 2) 
        return -1


    def dec_to_bin(self, dec_val):
        return bin(dec_val)[2:]


    def gen_scenary(self, n_layers):
        n_zeros = n_layers - 1
        start_value = '1'
        end_value = '1'

        start_value += '0' * n_zeros
        end_value += '1' * n_zeros

        scenary = []

        current_dec = self.bin_to_dec(start_value)
        scenary.append(start_value)

        first_dec = self.bin_to_dec(start_value)
        second_dec = first_dec

        for i in range(1, n_layers):
            first = i*'0' + self.dec_to_bin(int(first_dec/2))
            scenary.append(first)
            first_dec = self.bin_to_dec(first)

            second_dec = second_dec + first_dec
            second = self.dec_to_bin(second_dec)
            scenary.append(second)

        return scenary


    def gen_pass_scenary(self, n_layers):
        start = '0' + '1'*(n_layers-1)
        pass_scen = []

        pass_scen.append(start)
        start_dec = self.bin_to_dec(start)

        current = start_dec
        for i in range(1, n_layers-1):
            
            start = (i+1)*'0' + self.dec_to_bin(start_dec >> 1)

            pass_scen.append(start)
            pass_scen.append(start)

            start_dec = self.bin_to_dec(start)

        end = '0' * n_layers
        pass_scen.append(end)
        pass_scen.append(end)
        return pass_scen



