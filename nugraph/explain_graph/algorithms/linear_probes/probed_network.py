from nugraph.explain_graph.algorithms.linear_probes.linear_decoder import LinearDecoder
from nugraph.explain_graph.utils.load import Load
import torch 

class ProbedNetwork: 
    def __init__(self, 
                 model, 
                 planes=['u', 'v', 'y'], 
                 semantic_classes=['MIP','HIP','shower','michel','diffuse']
        ) -> None:
        
        self.model = model
        self.planes = planes

        self.semantic_classes = semantic_classes
        self.make_static_decoders()


    def make_static_decoders(self): 
        """
            Makes a decoder for each part of the network to turn the encoded space into something readable
        """

        input_inshape = self.model.encoder.net[self.planes[0]][0].net[0].weight.shape[1]

        encoder_inshape = self.model.encoder.net[self.planes[0]][0].net[0].weight.shape[0]
        planar_inshape = self.model.plane_net.net[self.planes[0]].node_net[-2].net[0].weight.shape[0]

        nexus_inshape = self.model.nexus_net.nexus_net[-2].net[0].weight.shape[0]
        
        linear_net = lambda input_shape: LinearDecoder(input_shape, self.planes, len(self.semantic_classes))

        self.input_decoder = linear_net((input_inshape, len(self.semantic_classes)))
        self.encoder_decoder =  linear_net(encoder_inshape) 
        self.planar_decoder =  linear_net(planar_inshape)
        self.nexus_decoder =  linear_net(nexus_inshape)

    def step_network(self, data, message_passing_steps=1, apply_softmax=False, apply_sigmoid=True): 
        """
            Iterate over the graph and perforce inference at each step using the static decoders
        """
        x, edge_index_plane, edge_index_nexus, nexus, batch = Load.unpack(data)
        forward = x.copy()
        input_decoded =  self.input_decoder.forward(forward)

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

        output = self.model.decoders[-1](forward, batch)['x_semantic']

        if apply_sigmoid: 
            input_decoded = self.input_decoder.probablity(input_decoded)
            encoder_decoded = self.encoder_decoder.probablity(encoder_decoded)
            planar_decoded = {key: self.planar_decoder.probablity(planar_decoded[key]) for key in planar_decoded.keys()}
            nexus_decoded ={key: self.nexus_decoder.probablity(nexus_decoded[key]) for key in nexus_decoded.keys()}
        
        if apply_softmax: 
            input_decoded = self.input_decoder.classes(input_decoded)
            encoder_decoded = self.encoder_decoder.classes(encoder_decoded)
            planar_decoded = {key: self.planar_decoder.classes(planar_decoded[key]) for key in planar_decoded.keys()}
            nexus_decoded ={key: self.nexus_decoder.classes(nexus_decoded[key]) for key in nexus_decoded.keys()}

        return input_decoded, encoder_decoded, planar_decoded, nexus_decoded, output


    def stepped_loss(self, input_decoded, encoder_decoded, planar_decoded, nexus_decoded, output): 
        return { 
            "input":self.loss(output, input_decoded), 
            "encoder_decoded": self.loss(output, encoder_decoded), 
            "planar_decoded": {i: self.loss(output, planar_decoded[i]) for i in planar_decoded.keys()}, 
            "nexus_decoded": {i: self.loss(output, nexus_decoded[i]) for i in nexus_decoded.keys()}, 

        } 

    def stepped_explaination(self, input_decoded, encoder_decoded, planar_decoded, nexus_decoded, output): 
        return { 
            "input_to_encoder": self.calculate_percent_explained(input_decoded, encoder_decoded), 
            "encoder_to_planar": self.calculate_percent_explained(encoder_decoded, planar_decoded[0]), 
            "planar_to_nexus": {i: self.calculate_percent_explained(planar_decoded[i], nexus_decoded[i]) for i in planar_decoded.keys()}, 
            "nexus_to_decoder": self.calculate_percent_explained(nexus_decoded[list(nexus_decoded.keys())[-1]], output)
        }


    def forward(self, data, message_passing_steps=1, apply_softmax=False): 
        network_probes = self.step_network(data, message_passing_steps, apply_softmax)
        return self.stepped_explaination(*network_probes), self.stepped_loss(*network_probes)
    
    def calculate_percent_explained(self, step1, step2): 
        entropy = {}
        for plane in self.planes: 
            step1_entropy = torch.distributions.categorical.Categorical(logits=step1[plane]).entropy().mean()
            step2_entropy = torch.distributions.categorical.Categorical(logits=step2[plane]).entropy().mean()
            entropy[plane]= (step1_entropy- step2_entropy).detach()
        return entropy
 
    def loss(self, y, y_hat): 
        loss = {}
        for plane in self.planes: 
            loss[plane] = torch.nn.CrossEntropyLoss()(y_hat[plane], y[plane]).item()#.detach()
        return loss