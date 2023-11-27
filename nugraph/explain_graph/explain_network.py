from explain_graph.load import Load
import torch 

class ExplainNetwork:
    """
        Abstract method for iterating a network and measuring a level of explaination for each iteration
        "Iteration" is a general term, can refer to message passing or stepping through layers. 
    """
    def __init__(self, checkpoint_path, data_path, batch_size, test, planes, semantic_classes) -> None:
        load = Load(checkpoint_path=checkpoint_path, data_path=data_path, batch_size=batch_size, test=test, planes=planes)
        self.planes = planes
        self.data = load.data
        self.model = load.model

        self.semantic_classes = semantic_classes

        self.static_decoder = self.make_static_decoders()

    def linear_decoder(self, in_shape): 
        num_classes = len(self.semantic_classes)
        class LinearDecoder: 
            def __init__(self, in_shape) -> None:
                self.decoder = []
                self.num_classes = num_classes
                for _ in range(num_classes):
                    self.decoder.append(torch.ones(in_shape, 1))
 
            def forward(self, X): 
                x = torch.tensor_split(X, self.num_classes, dim=1)
                return torch.cat([ net(x[i]) for i, net in enumerate(self.decoder) ], dim=1)

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
            return {plane:self.linear_decoder(in_shape) for plane in self.planes}

        return {
            "encoder": make_ones(),
            "planar":make_ones(),
            "nexus":make_ones(), 
            "decoder":[make_ones() for _ in self.model.decoders]
        }

    def step_network(self): 
        """
            Iterate over the graph and perforce inference at each step using the static decoders
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

    def step_network(self):
        return super().step_network()
    
    def calculate_percent_explained(self):
        """Using a entropy measure to see the information gain over the previous step"""
        return super().calculate_percent_explained()