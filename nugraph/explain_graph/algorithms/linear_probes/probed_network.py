from nugraph.explain_graph.algorithms.linear_probes.linear_decoder import LinearDecoder
from nugraph.explain_graph.utils.load import Load
import torch 

from nugraph.util import RecallLoss
from nugraph.explain_graph.algorithms.linear_probes.mutual_information import MutualInformation, DistributionEntropy


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

        linear_net = lambda input_shape: LinearDecoder(input_shape, self.planes, len(self.semantic_classes))

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