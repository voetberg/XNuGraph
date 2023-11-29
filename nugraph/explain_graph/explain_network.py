from nugraph.explain_graph.load import Load
import torch 

class ExplainNetwork:
    """
        Abstract method for iterating a network and measuring a level of explaination for each iteration
        "Iteration" is a general term, can refer to message passing or stepping through layers. 
    """
    def __init__(self, 
                 checkpoint_path=None, 
                 data_path=None, 
                 batch_size:int=1, 
                 test:bool=False, 
                 planes=['u','v','y'], 
                 semantic_classes = ['MIP','HIP','shower','michel','diffuse'],
) -> None:
        
        self.load = Load(checkpoint_path=checkpoint_path, data_path=data_path, batch_size=batch_size, test=test, planes=planes)
        self.planes = planes
        self.data = self.load.data
        self.model = self.load.model

        self.semantic_classes = semantic_classes
        self.make_static_decoders()

    def linear_decoder(self, in_shape): 
        num_classes = len(self.semantic_classes)
        planes = self.planes
        class LinearDecoder: 
            def __init__(self, in_shape) -> None:
                self.decoder = {}
                for plane in planes: 
                    self.decoder[plane] = self.class_decoder(in_shape)
 
            def class_decoder(self, in_shape): 
                decoder =  []
                for _ in range(num_classes):
                    decoder.append(torch.ones(in_shape, 1))
                return decoder 
            
            def class_decoder_forward(self, X, decoder): 
                x = torch.tensor_split(X, num_classes, dim=1) 
                return torch.cat([ net*x[i] for i, net in enumerate(decoder)], dim=1) 

            def forward(self, x): 
                return {
                    plane: self.class_decoder_forward(x[plane], self.decoder[plane]).squeeze(dim=-1) 
                    for plane in planes
                    } 
            
        return LinearDecoder(in_shape)

    def make_static_decoders(self): 
        """
            Makes a decoder for each part of the network to turn the encoded space into something readable
        """
        # Make a dictionary of decoders for the network - one for each part of the network (not including the decoders)
        # Assuming you can use Nugraph2 as a format guide

        # Static decoder acts as a step between the network and what the next step would be
        # - in is the same shape as the layer, but out is always the number of classes (acts as the semantic decoder)

        def make_ones(in_shape): 
            # Does the same thing as the linear layer, but there is no grad and it's all ones 
            return {plane: self.linear_decoder(in_shape) for plane in self.planes}
        
        input_inshape = self.model.encoder.net[self.planes[0]][0].net[0].weight.shape[1]
        input_inshape = int(input_inshape/len(self.planes))

        encoder_inshape = self.model.encoder.net[self.planes[0]][0].net[0].weight.shape[0]

        planar_inshape = self.model.plane_net.net[self.planes[0]].node_net[-2].net[0].weight.shape[0]

        nexus_inshape = 1

        self.input_decoder = make_ones(input_inshape)#self.linear_decoder(input_inshape) 
        self.encoder_decoder = self.linear_decoder(encoder_inshape)
        self.planar_decoder = self.linear_decoder(planar_inshape)
        self.nexus_decoder = self.linear_decoder(nexus_inshape)
        

    def step_network(self): 
        """
            Iterate over the graph and perforce inference at each step using the static decoders
        """
        x, edge_index_plane, edge_index_nexus, nexus, batch = self.load.unpack(self.data)
        forward = x.copy()
        input_decoded =  {plane: self.input_decoder[plane].forward(forward) for plane in self.planes}

        forward = self.model.encoder.forward(forward)
        encoder_decoded = self.encoder_decoder.forward(forward)

        # I wish it wouldn't but the plane net just modifies the input data instead of making a copy
        for p in self.planes:
            s = x[p].detach().unsqueeze(1).expand(-1, forward[p].size(1), -1)
            forward[p] = torch.cat((forward[p], s), dim=-1)
        
        self.model.plane_net(forward, edge_index_plane)
        planar_decoded = self.planar_decoder.forward(forward) 

        self.model.nexus_net(forward, edge_index_nexus, nexus)
        nexus_decoded = self.nexus_decoder.forward(forward) 

        output = {}
        for decoder in self.model.decoders:
            output.update(decoder(forward, batch))

        return input_decoded, encoder_decoded, planar_decoded, nexus_decoded, output

    def forward(self): 
        """
        """
        raise NotImplemented

    def calculate_percent_explained(self): 
        """
            Calculate how much each round of message passing through the network is impacting the final result 
        """
        raise NotImplemented
    

class ExplainMessages(ExplainNetwork): 
    def __init__(self, checkpoint_path, data_path, batch_size, test, planes) -> None:
        super().__init__(checkpoint_path, data_path, batch_size, test, planes)

    
    def calculate_percent_explained(self):
        """Using a entropy measure to see the information gain over the previous step"""
        return super().calculate_percent_explained()