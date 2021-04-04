
import torch

class Trainer():
    def __init__(self, net, cfg, optimizer, criterion):
        self.net = net
        self.cfg = cfg
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = cfg.epochs

        self.scenary = self.gen_scenary(cfg.layers)
        self.pass_scenary = self.gen_pass_scenary(cfg.layers)
    

    def train_with_scenary(self, u, h_data_noisy, data_noise_power, scen0):
        for i in range(0, len(self.scenary)):
            print('\n', 'Stage ', i)
            self.net.setState(self.scenary[i], self.pass_scenary[i])
            self.train(tr_round = i, u = u, h_data_noisy= h_data_noisy, data_noise_power = data_noise_power, scen0= scen0, epochs = self.epochs)


    def train(self, u , tr_round, h_data_noisy, data_noise_power, scen0, epochs):
        degrade_counter = 0
        flat_counter = 0

        prev_loss = 0
        for i in range(epochs):
            # зануляем градиент
            self.optimizer.zero_grad()
            
            #прогоняем через сетку
            h_rec = self.net.forward(u)

            loss_value = self.criterion(h_rec, h_data_noisy, data_noise_power, scen0) 
            #MSE_detector_loss(h_rec)

            print('round = ',tr_round,', loss = ',loss_value)

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
