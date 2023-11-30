from wisp.models.latent_decoders.basic_latent_decoder import *
from torch.nn.modules.utils import _ntuple


class MultiSequential(nn.Sequential):
    def forward(self, input, alpha):
        for module in self._modules.values():
            if isinstance(module,MultiLatentDecoderLayer):
                input = module(input, alpha)
                # print(input.min(),input.max())
            else:
                input = module(input)
        return input

class StraightThroughOneHot(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        out = torch.argmax(x,dim=0)
        out = torch.nn.functional.one_hot(out,num_classes=x.size(0)).permute(1,0).to(x)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class MultiLatentDecoderLayer(Module):

    def __init__(self, in_features: int, out_features: int, ldecode_matrix: str, num_decoders: int = 1, bias: bool = False) -> None:
        super(MultiLatentDecoderLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if 'dft' in ldecode_matrix:
            self.dft = Parameter(get_dft_matrix(in_features, out_features), requires_grad=False)
        if 'dft' in ldecode_matrix:
            self.scale = Parameter(torch.empty((num_decoders,1,out_features)))
        else:
            self.scale = Parameter(torch.empty((num_decoders,in_features,out_features)))
        if bias:
            self.use_shift = Parameter(torch.empty(num_decoders,1,out_features))
        else:
            self.register_parameter('use_shift', None)

        self.ldecode_matrix = ldecode_matrix
        if ldecode_matrix == 'dft_fixed':
            self.scale.requires_grad_(False)
            if not bias:
                self.use_shift.requires_grad_(False)

    def reset_parameters(self, param=1.0, init_type = 'normal') -> None:
        if init_type == 'normal':
            init.normal_(self.scale, std=param)
        elif init_type == 'uniform':
            init.uniform_(self.scale, -param, param)
        elif init_type == 'constant':
            init.constant_(self.scale, val=param)
        if self.use_shift is not None:
            init.zeros_(self.use_shift)

    def clamp(self, val: float = 0.5) -> None:
        with torch.no_grad():
            self.scale.clamp_(-val, val)

    def forward(self, input: Tensor, alpha: Tensor) -> Tensor:

        if 'dft' in self.ldecode_matrix:
            out = torch.matmul(input,self.dft).unsqueeze(0)
            w_out = out*self.scale+(self.use_shift if self.use_shift is not None else 0)
        else:
            # out = torch.matmul(input.unsqueeze(1).unsqueeze(0),self.scale.unsqueeze(1)).squeeze(2)
            out = torch.empty((self.scale.size(0),*input.size()),dtype=input.dtype,device=input.device)
            for i in range(self.scale.size(0)):
                out[i] = torch.matmul(input,self.scale[i])
            out = torch.sum(out*alpha.unsqueeze(-1),dim=0)
            w_out = out+(self.use_shift if self.use_shift is not None else 0)
        w_out = torch.sum(w_out*alpha.unsqueeze(-1),dim=0)
        return w_out

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.use_shift is not None
        )

class MultiLatentDecoder(Module):

    def __init__(
        self,
        latent_dim: int,
        feature_dim: int,
        norm: str,
        ldecode_matrix:str,
        use_shift: bool,
        num_entries:int,
        num_layers_dec:int = 0,
        hidden_dim_dec:int = 0,
        activation:str = 'none',
        final_activation:str = 'none',
        clamp_weights:float = 0.0,
        ldec_std:float = 1.0,
        num_decoders:int = 1,
        alpha_std: float = 1.0,
        use_sga: bool = False,
        **kwargs,
    ) -> None:
        super(MultiLatentDecoder, self).__init__()
        latent_dim = feature_dim if latent_dim == 0 else latent_dim
        self.ldecode_matrix = ldecode_matrix
        self.channels = feature_dim
        self.latent_dim = latent_dim
        self.norm = norm
        self.div = nn.Parameter(torch.ones(latent_dim),requires_grad=False)
        self.num_layers_dec =  num_layers_dec
        if num_layers_dec>0:
            if hidden_dim_dec == 0:
                hidden_dim_dec = feature_dim
            self.hidden_dim_dec = _ntuple(num_layers_dec)(hidden_dim_dec)
        self.use_shift = use_shift
        
        act_dict = {
                    'none':nn.Identity(), 
                    'sigmoid':nn.Sigmoid(), 
                    'tanh':nn.Tanh(),
                    'relu':nn.ReLU(), 
                    'sine':SineScaled(30.0)
                    }
        self.act = act_dict[activation]
        self.final_activation = act_dict[final_activation]
        self.clamp_weights = clamp_weights
        self.num_decoders = num_decoders

        layers = []
        for l in range(num_layers_dec):
            feature_dim = self.hidden_dim_dec[l]
            feature_dim = latent_dim if feature_dim == 0 else feature_dim
            layers.append(MultiLatentDecoderLayer(latent_dim, feature_dim, ldecode_matrix, num_decoders=num_decoders, bias=self.use_shift))
            layers.append(self.act)
            latent_dim = feature_dim
        feature_dim = self.channels
        layers.append(MultiLatentDecoderLayer(latent_dim,feature_dim,ldecode_matrix, num_decoders=num_decoders,bias=self.use_shift))

        self.alpha = nn.Parameter(torch.randn(num_decoders,num_entries)*alpha_std, requires_grad=True)

        self.temperature = 1.0
        self.layers = MultiSequential(*layers)
        self.reset_parameters('normal', ldec_std)
        self.straight_through = True
        self.use_sga = use_sga
        self.diff_sampling = False
        
    def reset_parameters(self, init_type, param=0.5) -> None:
        for layer in list(self.layers.children()):
            if isinstance(layer, MultiLatentDecoderLayer):
                layer.reset_parameters(param,init_type)

    def get_scale(self):
        assert self.num_layers_dec == 0, "Can only get scale for 0 hidden layers decoder!"
        return list(self.layers.children())[0].scale

    def clamp(self, val: float = 0.2) -> None:
        for layer in list(self.layers.children()):
            if isinstance(layer, MultiLatentDecoderLayer):
                layer.clamp(val)

    def size(self, use_torchac=False):
        fp_size = sum([p.numel()*torch.finfo(p.dtype).bits for n,p in self.named_parameters() if 'alpha' not in n])
        alpha = torch.argmax(self.alpha,dim=0)
        if not use_torchac:
            unique_vals, counts = torch.unique(alpha, return_counts = True)
            probs = counts/torch.sum(counts)
            information_bits = torch.clamp(-1.0 * torch.log(probs + 1e-10) / np.log(2.0), 0, 1000)
            size_bits = torch.sum(information_bits*counts).item()
            return size_bits+fp_size
        else:
            import torchac
            alpha = alpha - alpha.min()
            unique_vals, counts = torch.unique(alpha, return_counts = True)
            mapping = torch.zeros((alpha.max().item()+1))
            mapping[unique_vals] = torch.arange(unique_vals.size(0)).to(mapping)
            alpha = mapping[alpha]
            cdf = torch.cumsum(counts/counts.sum(),dim=0)
            cdf = torch.cat((torch.Tensor([0.0]),cdf))
            cdf = cdf.unsqueeze(0).repeat(alpha.size(0),1)
            cdf = cdf/cdf[:,-1:] # Normalize the final cdf value just to keep torchac happy
            byte_stream = torchac.encode_float_cdf(cdf.detach().cpu(), alpha.detach().cpu().to(torch.int16), \
                                                    check_input_bounds=True)
            return len(byte_stream)*8+fp_size

    def forward(self, weight: Tensor) -> Tensor:

        alpha = nn.functional.softmax(self.alpha/self.temperature,dim=0)
        if self.straight_through:
            alpha = StraightThroughOneHot.apply(alpha)

        if self.use_sga:
            weightf = torch.floor(weight) if self.diff_sampling else StraightThroughFloor.apply(weight)
            weightc = weightf+1
            logits_f = -torch.tanh(torch.clamp(weight-weightf, min=-1+epsilon, max=1-epsilon)).unsqueeze(-1)/self.temperature
            logits_c = -torch.tanh(torch.clamp(weightc-weight, min=-1+epsilon, max=1-epsilon)).unsqueeze(-1)/self.temperature
            logits = torch.cat((logits_f,logits_c),dim=-1)
            dist = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(self.temperature, logits=logits)
            sample = dist.rsample() if self.diff_sampling else dist.sample()
            weight = weightf*sample[...,0]+weightc*sample[...,1]
        else:
            weight = StraightThrough.apply(weight)
        w_out = self.layers(weight/self.div, alpha)
        w_out = self.final_activation(w_out)
        if self.clamp_weights>0.0:
            w_out = torch.clamp(w_out, min=-self.clamp_weights, max=self.clamp_weights)

        return w_out
