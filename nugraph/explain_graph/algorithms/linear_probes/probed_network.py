import math
import torch 
import numpy as np
import matplotlib.pyplot as plt

import json 
from tqdm import tqdm

from nugraph.explain_graph.algorithms.linear_probes.linear_decoder import StaticLinearDecoder, DynamicLinearDecoder
from nugraph.explain_graph.utils.load import Load
from nugraph.util import RecallLoss
from nugraph.explain_graph.algorithms.linear_probes.mutual_information import MutualInformation
from nugraph.explain_graph.algorithms.linear_probes.feature_loss import FeatureLoss

class ProbedNetwork: 
    def __init__(self, 
                 model, 
                 planes=['u', 'v', 'y'], 
                 semantic_classes=['MIP','HIP','shower','michel','diffuse'], 
                 explain_metric = MutualInformation(), # DistributionEntropy(), 
                 loss_metric = RecallLoss(), 
        ) -> None:
        
        self.model = model
        self.planes = planes

        self.semantic_classes = semantic_classes
        self.make_static_decoders()
        
        self.explain_metric = explain_metric
        self.loss_metric = loss_metric

    def make_static_decoders(self): 
        """
            Makes a decoder for each part of the network to turn the encoded space into something readable
        """

        input_inshape = self.model.encoder.net[self.planes[0]][0].net[0].weight.shape[1]
        encoder_inshape = self.model.encoder.net[self.planes[0]][0].net[0].weight.shape[0]
        planar_inshape = self.model.plane_net.net[self.planes[0]].node_net[-2].net[0].weight.shape[0]

        decoder_inshape = len(self.semantic_classes)#self.model.decoders[0].net[self.planes[0]].net[0].weight.shape[-1]

        linear_net = lambda input_shape: StaticLinearDecoder(input_shape, self.planes, len(self.semantic_classes))

        self.input_decoder = linear_net((input_inshape, len(self.semantic_classes)))
        self.encoder_decoder =  linear_net((encoder_inshape, 1))#len(self.semantic_classes)))
        self.planar_decoder =  linear_net((planar_inshape,1))
        self.nexus_decoder =  linear_net((planar_inshape,1))
        self.output_decoder = linear_net((decoder_inshape,len(self.semantic_classes)))

    def step_network(self, data, message_passing_steps=1, apply_softmax=False, apply_sigmoid=True): 
        """
            Iterate over the graph and perforce inference at each step using the static decoders
        """
        x, edge_index_plane, edge_index_nexus, nexus, batch = Load.unpack(data)
        x = {plane: x[plane][:,:4] for plane in self.planes}
        forward = x.copy()
        input_decoded =  self.input_decoder.forward(forward)
        #input_decoded = {plane: input_decoded[plane].unsqueeze(-1) for plane in self.planes}

        forward = self.model.encoder.forward(forward)
        encoder_decoded = self.encoder_decoder.forward(forward)
        # I wish it wouldn't but the plane net just modifies the input data instead of making a copy

        planar_decoded = {}
        nexus_decoded = {}

        for step in range(message_passing_steps): 

            for p in self.planes:
                s = x[p].detach().unsqueeze(1).expand(-1, forward[p].size(1), -1)
                forward[p] = torch.cat((forward[p], s), dim=-1)

            self.model.plane_net(forward, edge_index_plane)
            planar_decoded[step] = self.planar_decoder.forward(forward) 

            self.model.nexus_net(forward, edge_index_nexus, nexus)
            nexus_decoded[step] = self.nexus_decoder.forward(forward) 

        output = self.model.decoders[0](forward, batch)['x_semantic']
        output_decoded = self.output_decoder.forward(output)

        if apply_sigmoid: 
            input_decoded = self.input_decoder.probablity(input_decoded)
            encoder_decoded = self.encoder_decoder.probablity(encoder_decoded)
            planar_decoded = {key: self.planar_decoder.probablity(planar_decoded[key]) for key in planar_decoded.keys()}
            nexus_decoded ={key: self.nexus_decoder.probablity(nexus_decoded[key]) for key in nexus_decoded.keys()}
            output_decoded = self.output_decoder.probablity(output_decoded)
        
        if apply_softmax: 
            input_decoded = self.input_decoder.classes(input_decoded)
            encoder_decoded = self.encoder_decoder.classes(encoder_decoded)
            planar_decoded = {key: self.planar_decoder.classes(planar_decoded[key]) for key in planar_decoded.keys()}
            nexus_decoded ={key: self.nexus_decoder.classes(nexus_decoded[key]) for key in nexus_decoded.keys()}
            output_decoded = self.output_decoder.classes(output_decoded)

        return input_decoded, encoder_decoded, planar_decoded, nexus_decoded, output_decoded, output


    def stepped_loss(self, input_decoded, encoder_decoded, planar_decoded, nexus_decoded, output_decoded, output): 
        return { 
            "input":self.loss(output, input_decoded), 
            "encoder_decoded": self.loss(output, encoder_decoded), 
            "planar_decoded": {i: self.loss(output, planar_decoded[i]) for i in planar_decoded.keys()}, 
            "nexus_decoded": {i: self.loss(output, nexus_decoded[i]) for i in nexus_decoded.keys()}, 
            "output_decoded": self.loss(output, output_decoded)

        } 

    def stepped_explaination(self, input_decoded, encoder_decoded, planar_decoded, nexus_decoded, output_decoded): 
        return { 
            "input_to_encoder": self.calculate_percent_explained(input_decoded, encoder_decoded), 
            "encoder_to_planar": self.calculate_percent_explained(encoder_decoded, planar_decoded[0]), 
            "planar_to_nexus": {i: self.calculate_percent_explained(planar_decoded[i], nexus_decoded[i]) for i in planar_decoded.keys()}, 
            "nexus_to_planar":{i: self.calculate_percent_explained(nexus_decoded[i], planar_decoded[i]) for i in planar_decoded.keys()}, 
            "nexus_to_decoder": self.calculate_percent_explained(nexus_decoded[list(nexus_decoded.keys())[-1]], output_decoded)
        }


    def forward(self, data, message_passing_steps=1, apply_softmax=False): 
        network_probes = self.step_network(data, message_passing_steps, apply_softmax)
        return self.stepped_explaination(*network_probes[:-1]), self.stepped_loss(*network_probes)
    
    def calculate_percent_explained(self, step1, step2): 
        entropy = {}
        for plane in self.planes: 
            entropy[plane]= self.explain_metric(step1[plane], step2[plane]).item()
        return entropy
 
    def loss(self, y, y_hat): 
        loss = {}
        for plane in self.planes: 
            loss[plane] = self.loss_metric(y_hat[plane], y[plane]).item()
        return loss
    

class DynamicProbedNetwork(ProbedNetwork): 

    def __init__(self, 
                 model, 
                 data,
                 planes=['u', 'v', 'y'], 
                 semantic_classes=['MIP','HIP','shower','michel','diffuse'], 
                 explain_metric = MutualInformation(), 
                 loss_metric = RecallLoss(), 
                 epochs:int = 25, 
                 message_passing_steps=5,
                 out_path="./"
        ) -> None:
        super().__init__(model, planes, semantic_classes, explain_metric, loss_metric)
        self.message_passing_steps = message_passing_steps
        self.epochs = epochs
        self.probe_training_history = {}
        self.make_probes()

    def make_probes(self): 
        probe = lambda in_shape: DynamicLinearDecoder((in_shape, 1), self.planes, len(self.semantic_classes)).cuda()
        train_probe_track = lambda probe: TrainProbes(probe, loss_function="tracks")
        train_probe_hipmip = lambda probe: TrainProbes(probe, loss_function="hipmip")

        encoder_inshape = self.model.encoder.net[self.planes[0]][0].net[0].weight.shape[0]
        planar_inshape = self.model.plane_net.net[self.planes[0]].node_net[-2].net[0].weight.shape[1]

        encoder_decoder =  probe(encoder_inshape)
        self.encoder_training_track = train_probe_track(encoder_decoder)
        encoder_decoder =  probe(encoder_inshape)
        self.encoder_training_hipmip = train_probe_hipmip(encoder_decoder)

        planar_decoder = [probe(planar_inshape) for _ in range(self.message_passing_steps)]
        self.planar_training_track = [train_probe_track(decoder) for decoder in planar_decoder]
        planar_decoder = [probe(planar_inshape) for _ in range(self.message_passing_steps)]
        self.planar_training_hipmip = [train_probe_hipmip(decoder) for decoder in planar_decoder]

        nexus_decoder =  [probe(planar_inshape) for _ in range(self.message_passing_steps)]
        self.nexus_training_track = [train_probe_track(decoder) for decoder in nexus_decoder]
        nexus_decoder =  [probe(planar_inshape) for _ in range(self.message_passing_steps)]
        self.nexus_training_hipmip = [train_probe_hipmip(decoder) for decoder in nexus_decoder]

        output_decoder = probe(planar_inshape)
        self.output_training_track = train_probe_track(output_decoder)
        output_decoder = probe(planar_inshape)
        self.output_training_hipmip = train_probe_hipmip(output_decoder)

    def step_network_batch(self, batch, message_passing_steps): 
        batch_history = { 
            "track": {"encoder":{}, "planar":{}, "nexus":{}, "output":{}}, 
            "hipmip": {"encoder":{}, "planar":{}, "nexus":{}, "output":{}}
        }
        x, edge_index_plane, edge_index_nexus, nexus, _ = Load.unpack(batch)
        x = {plane: x[plane][:,:4] for plane in self.planes}
        forward = x.copy()

        forward = self.model.encoder.forward(forward)

        encoder_forward = forward.copy()
        train, _ = self.encoder_training_track.step([encoder_forward], [batch])
        batch_history['track']["encoder"] = train
        train, _ = self.encoder_training_hipmip.step([encoder_forward], [batch])
        batch_history['hipmip']["encoder"] = train

        planar_decoded_track = []
        nexus_decoded_track = []

        planar_decoded_hm = []
        nexus_decoded_hm = []

        for message_step in range(message_passing_steps): 
        
            for p in self.planes:
                s = x[p].detach().unsqueeze(1).expand(-1, forward[p].size(1), -1)
                forward[p] = torch.cat((forward[p], s), dim=-1)

            self.model.plane_net(forward, edge_index_plane)
            planar_forward = forward.copy() 

            train, _ = self.planar_training_track[message_step].step([planar_forward], [batch])
            planar_decoded_track.append(train)
            train, _ = self.planar_training_hipmip[message_step].step([planar_forward], [batch])
            planar_decoded_hm.append(train)

            self.model.nexus_net(forward, edge_index_nexus, nexus)
            nexus_forward = forward.copy() 

            train, _ = self.nexus_training_track[message_step].step([nexus_forward], [batch])
            nexus_decoded_track.append(train)
            train, _ = self.nexus_training_hipmip[message_step].step([nexus_forward], [batch])
            nexus_decoded_hm.append(train)


        batch_history['track']["planar"] = planar_decoded_track
        batch_history['hipmip']["planar"] = planar_decoded_hm
        batch_history['track']["nexus"] = nexus_decoded_track
        batch_history['hipmip']["nexus"] = nexus_decoded_hm

        output = self.model.decoders[0](forward, batch)['x_semantic']
        output_forward = output.copy()
        output_forward = {
            plane: torch.stack([
                output_forward[plane] for _ in range(nexus_forward[plane].shape[-1])
                ]).swapaxes(0, -1).swapaxes(0, 1)
            for plane in self.planes}
        
        train, _ = self.output_training_track.step([output_forward],[batch])
        batch_history['track']["output"] = train
        train, _ = self.output_training_hipmip.step([output_forward],[batch])
        batch_history['hipmip']["output"] = train

        return batch_history

    def add_history(self, batch_history): 
        for key in self.probe_history_tracks: 
            if key in ['planar', 'nexus']: 
                for message_step in range(len(batch_history['track'][key])): 
                    self.probe_history_tracks[key][-1][message_step]+=batch_history["track"][key][message_step]
                    self.probe_history_hipmip[key][-1][message_step]+=batch_history["hipmip"][key][message_step]

            else: 
                self.probe_history_tracks[key][-1]+=batch_history["track"][key]
                self.probe_history_hipmip[key][-1]+=batch_history["hipmip"][key]

    def update_history_index(self, n_batches:int): 
        for key in self.probe_history_tracks: 
            if key in ['planar', 'nexus']: 
                for message_step in range(self.message_passing_steps): 
                    self.probe_history_tracks[key][-1][message_step] /= n_batches
                    self.probe_history_hipmip[key][-1][message_step] /= n_batches

                self.probe_history_tracks[key].append([0 for _ in range(self.message_passing_steps)])
                self.probe_history_hipmip[key].append([0 for _ in range(self.message_passing_steps)])

            else: 
                self.probe_history_tracks[key][-1] /= n_batches
                self.probe_history_hipmip[key][-1] /= n_batches

                self.probe_history_tracks[key].append(0)
                self.probe_history_hipmip[key].append(0)

    def step_network_with_training(self, data, message_passing_steps, out_path): 
        self.probe_history_tracks = {
                "encoder":[0],
                "planar": [[0 for _ in range(message_passing_steps)]], 
                "nexus":  [[0 for _ in range(message_passing_steps)]],
                "output": [0]
            }
        
        self.probe_history_hipmip = {
                "encoder":[0],
                "planar": [[0 for _ in range(message_passing_steps)]], 
                "nexus":  [[0 for _ in range(message_passing_steps)]],
                "output": [0]
            }
        
        for _ in tqdm(range(self.epochs)): 
            for batch in data: 
                batch_history = self.step_network_batch(batch, message_passing_steps)
                self.add_history(batch_history)

            self.update_history_index(len(data))
            with open(f"{out_path}/track_trainer_history.json", 'w') as f: 
                json.dump(self.probe_history_tracks, f)
            with open(f"{out_path}/hipmip_trainer_history.json", 'w') as f: 
                json.dump(self.probe_history_hipmip, f)
        

    def plot_probe_training_history(self, out_path, file_name=""): 
        plt.close("all")
        keys = self.probe_history_tracks.keys()
        fig, subplots = plt.subplots(nrows=2, ncols=len(keys), sharey=True, sharex=False, figsize=(5*len(keys), 10))
        
        ylabels = ["Track identification", "Hip/Mip difference"]
        for subplot, key in enumerate(keys): 
            for col, history in enumerate([self.probe_history_tracks, self.probe_history_hipmip]):
                train = history[key]
                index = [i for i in range(len(train))]
                # handle the message passing ones 

                if type(train[0]) == list: 
                    train = np.array(train).T

                    for message_index, message_step in enumerate(train): 
                        index = [i for i in range(len(message_step))]
                        subplots[col, subplot].plot(index, message_step, label=f"Message Step {message_index+1}")

                    subplots[col, subplot].legend()
                else: 
                    subplots[col, subplot].plot(index, train, label='Train', color="blue")

                subplots[col, subplot].set_title(key)
                subplots[col, 0].set_ylabel(ylabels[col])

        fig.supxlabel("Training Epoch")
        fig.supylabel("Loss")
        fig.tight_layout() 
        plt.legend()
        plt.savefig(f"{out_path.rstrip('/')}/{file_name}_probe_loss.png")

    def forward(self, data, out_path="./"):
        self.probe_history_tracks = {
                "encoder":[0],
                "planar": [[0 for _ in range(self.message_passing_steps)]], 
                "nexus":  [[0 for _ in range(self.message_passing_steps)]],
                "output": [0]
        }
        
        self.probe_history_hipmip = {
                "encoder":[0],
                "planar": [[0 for _ in range(self.message_passing_steps)]], 
                "nexus":  [[0 for _ in range(self.message_passing_steps)]],
                "output": [0]
            }
        
        for _ in tqdm(range(self.epochs)): 
            for batch in data: 
                batch_history = self.step_network_batch(batch, self.message_passing_steps)
                self.add_history(batch_history)

            self.update_history_index(len(data))
            with open(f"{out_path}/track_trainer_history.json", 'w') as f: 
                json.dump(self.probe_history_tracks, f)
            with open(f"{out_path}/hipmip_trainer_history.json", 'w') as f: 
                json.dump(self.probe_history_hipmip, f)

    def stepped_explaination(self, input_decoded, encoder_decoded, planar_decoded, nexus_decoded, output_decoded):
        return {}, {} 
    
class TrainProbes: 
    def __init__(self, probe:DynamicLinearDecoder, loss_function:str, planes:list=['v', 'u', 'y']) -> None:
        self.probe = probe 
        self.probe_loss_train = {}
        self.probe_loss_validation = {}
        self.loss_function = FeatureLoss(loss_function).loss
        self.planes = planes 

        self.optimizer = torch.optim.SGD(params = self.probe.decoder.parameters(), lr=0.01)

    def loss(self, x, labels): 
        #labels = Batch.from_data_list([datum for datum in labels])
        prediction = self.probe.forward(x[0])
        loss = self.loss_function(prediction, labels[0])
        return loss

    def train(self, forward_data, labels):
        self.probe.train(True)
        loss = self.loss(forward_data, labels)
        for plane in loss: 
            plane.backward(retain_graph=True)
        self.optimizer.step()
        loss = torch.stack(loss).mean().item()
        return loss

    def step(self, forward_data, labels): 
        train_loss = self.train(forward_data, labels)
        return train_loss, 0 