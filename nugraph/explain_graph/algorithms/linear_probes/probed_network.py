import torch 
import tqdm
from typing import Any, Callable
import matplotlib.pyplot as plt

from nugraph.explain_graph.algorithms.linear_probes.linear_decoder import StaticLinearDecoder, DynamicLinearDecoder
from nugraph.explain_graph.utils.load import Load
from nugraph.util import RecallLoss
from nugraph.explain_graph.algorithms.linear_probes.mutual_information import MutualInformation


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
        #output_decoded = {plane: output_decoded[plane].unsqueeze(-1) for plane in self.planes}

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
        ) -> None:
        super().__init__(model, planes, semantic_classes, explain_metric, loss_metric)
        self.data = data
        self.make_probes()

    def make_probes(self): 
        probe = lambda in_shape: DynamicLinearDecoder(in_shape, self.planes, len(self.semantic_classes))
        train_probe = lambda probe: TrainProbes(probe, loss_function=self.loss_metric, data=self.data)()

        input_inshape = ""
        encoder_inshape = ""
        planar_inshape = ""
        decoder_inshape = ""

        self.input_decoder = probe(input_inshape)
        input_train, input_val = train_probe(self.input_decoder)

        self.encoder_decoder =  probe((encoder_inshape, 1))#len(self.semantic_classes)))
        encoder_train, encoder_val = train_probe(self.encoder_decoder)

        # TODO Multiple probes for different message passing steps 
        self.planar_decoder =  probe((planar_inshape,1))
        planar_train, planar_val = train_probe(self.planar_decoder)

        self.nexus_decoder =  probe((planar_inshape,1))
        nexus_train, nexus_val = train_probe(self.nexus_decoder)

        self.output_decoder = probe((decoder_inshape,len(self.semantic_classes)))
        output_train, output_val = train_probe(self.output_decoder)

        self.probe_training_history = {
            "train": {
                "input": input_train, 
                "encoder": encoder_train, 
                "planar": planar_train, 
                "nexus": nexus_train, 
                "output": output_train
                      },
            "val": {
                "input": input_val, 
                "encoder": encoder_val, 
                "planar": planar_val, 
                "nexus": nexus_val, 
                "output": output_val
            }
        }

    def plot_probe_training_history(self, out_path, file_name=""): 

        plt.close("all")

        fig, subplots = plt.subplots(nrows=1, ncols=5)
        for subplot, key in zip(subplots, self.probe_training_history['train'].keys()): 
            
            train = self.probe_training_history['train'][key].values()
            val = self.probe_training_history['train'][key].values()
            index = self.probe_training_history['train'][key].keys()

            subplot.plot(index, train, label='Train', color="blue")
            subplot.plot(index, val, label="Val", linestyle=(5, (10, 3)), color="orange")
            subplot.set_title(key)

        fig.subpxlabel("Training Epoch")
        fig.supylabel("Loss")
        fig.tight_layout() 
        plt.legend()
        plt.savefig(f"{out_path.rstrip('/')}/{file_name}_probe_loss.png")


class TrainProbes: 
    def __init__(self, probe:DynamicLinearDecoder, loss_function:Callable, data) -> None:
        self.probe = probe 
        self.probe_loss_train = {}
        self.probe_loss_validation = {}
        self.data = data
        self.loss_function = loss_function

        self.optimizer = torch.optim.SGD(params = self.probe.params, lr=0.01)

    def loss(self, x, labels): 
        prediction = self.probe.forward(x)
        loss = self.loss_function(prediction, labels)
        return loss

    def train(self):
        self.probe.train(True)
        running_loss = []
        for batch in self.data.train: 
            loss = self.loss(batch, labels=batch[""])
            loss.backward()
            self.optimizer.step()
            running_loss.append(loss)

        loss = torch.mean(torch.tensor(running_loss))
        return loss
    
    def validate(self): 
        self.probe.train(False)
        running_loss = []
        for batch in self.data.validation: 
            loss = self.loss(batch, labels=batch[""])
            running_loss.append(loss)

        loss = torch.mean(torch.tensor(running_loss))
        return loss

    def __call__(self, epochs=20) -> Any:
        for epoch in tqdm.tqdm(range(epochs)): 
            train_loss = self.train()
            val_loss = self.validate()
            self.probe_loss_train[epoch] = train_loss
            self.probe_loss_validation[epoch] = val_loss

        return self.probe_loss_train, self.probe_loss_validation