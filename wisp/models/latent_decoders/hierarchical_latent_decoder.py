from wisp.models.latent_decoders.basic_latent_decoder import *

class HierarchicalLatentDecoder(nn.Module):

    def __init__(self, num_decoders, offsets, conf_decoder):
        super().__init__()
        self.num_decoders = num_decoders
        self.decoders = nn.ModuleList([LatentDecoder(**conf_decoder) for _ in range(self.num_decoders)])
        self.offsets = offsets

    def forward(self, input):
        output = torch.empty((input.size(0),self.decoders[0].channels)).to(input)
        for l in range(self.num_decoders):
            output[self.offsets[l]:self.offsets[l+1]] = self.decoders[l](input[self.offsets[l]:self.offsets[l+1]])
        return output

    @property
    def temperature(self):
        return self.decoders[0].temperature
    
    @property
    def use_sga(self):
        return self.decoders[0].use_sga
    
    @temperature.setter
    def temperature(self, value):
        for l in range(self.num_decoders):
            self.decoders[l].temperature = value

    @use_sga.setter
    def use_sga(self, value):
        for l in range(self.num_decoders):
            self.decoders[l].use_sga = value
            
    def size(self, use_torchac=False):
        return sum([p.numel()*torch.finfo(p.dtype).bits for p in self.parameters()])