import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet18_encoder import *
from models.resnet20_cifar import *

class MYNET(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode
        self.args = args
        if self.args.dataset in ['cifar100']:
            self.encoder = resnet20()
            self.num_features = 64
        if self.args.dataset in ['mini_imagenet']:
            self.encoder = resnet18(False, args)  # pretrained=False
            self.num_features = 512
        if self.args.dataset in ['cub200']:
            self.encoder = resnet18(True, args)  # pretrained=True follow TOPIC, models for cub is imagenet pre-trained. https://github.com/xyutao/fscil/issues/11#issuecomment-687548790
            self.num_features = 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_real = nn.Linear(self.num_features, self.args.num_classes, bias=False)

        self.fc_auxs = nn.ModuleList([
            nn.Linear(self.num_features, self.args.way, bias=False)
            for _ in range(self.args.sessions-1)
        ])

        self.attentions = nn.ModuleList([
            SelfAttention(self.num_features) for _ in range(self.args.sessions-1)
        ])

    def forward_metric(self, x):
        map = self.encoder(x)
        feat = self.avgpool(map)
        feat = feat.squeeze(-1).squeeze(-1)

        x = F.linear(F.normalize(feat, p=2, dim=-1), F.normalize(self.fc_real.weight, p=2, dim=-1))
        x = self.args.temperature * x

        return x, feat, map

    def get_inc_output(self, images, session):
        encoder_feature = self.encoder(images)
        encoder_feature = self.attentions[session-1](encoder_feature)
        encoder_feature = self.avgpool(encoder_feature)
        encoder_feature = encoder_feature.squeeze(-1).squeeze(-1)
        wf = F.linear(F.normalize(encoder_feature, p=2, dim=1), F.normalize(self.fc_auxs[session-1].weight, p=2, dim=1))
        return wf

    def encode(self, x):
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x

    def forward(self, input):
        if self.mode != 'encoder':
            input = self.forward_metric(input)
            return input
        elif self.mode == 'encoder':
            input = self.encode(input)
            return input
        else:
            raise ValueError('Unknown mode')

    def update_fc(self,dataloader,class_list):
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            data=self.encode(data).detach()

        if self.args.not_data_init:
            new_fc = nn.Parameter(
                torch.rand(len(class_list), self.num_features, device="cuda"),
                requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            new_fc = self.update_fc_avg(data, label, class_list)

        corr_matrix = torch.zeros(new_fc.size(0), new_fc.size(0))
        for i in range(new_fc.size(0)):
            for j in range(new_fc.size(0)):
                corr_matrix[i][j] = self.RS(new_fc[i], new_fc[j])

        return corr_matrix

    def deviations(self, x, y):
        mean_x = torch.mean(x)
        mean_y = torch.mean(y)
        xm = x.sub(mean_x)
        ym = y.sub(mean_y)
        r_num = xm.dot(ym)
        r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
        r_val = r_num / r_den
        return r_val

    def RS(self, x, y):
        cosine_sim = torch.nn.functional.cosine_similarity(x, y, dim=0)
        pearson_val = self.deviations(x, y)
        return cosine_sim * pearson_val

    def update_fc_avg(self,data,label,class_list):
        new_fc=[]
        for class_index in class_list:
            data_index=(label==class_index).nonzero().squeeze(-1)
            embedding=data[data_index]
            proto=embedding.mean(0)
            new_fc.append(proto)
            self.fc_real.weight.data[class_index]=proto
        new_fc=torch.stack(new_fc,dim=0)
        return new_fc

    def get_logits(self,x,fc):
        if 'dot' in self.args.new_mode:
            return F.linear(x,fc)
        elif 'cos' in self.args.new_mode:
            return self.args.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))

    def count_params(self, session=None):
        """Count parameters for the entire model or specific session"""
        if session is None:
            # Count total parameters
            return sum(p.numel() for p in self.parameters())
        else:
            # Count parameters for specific session
            params = 0
            for name, p in self.named_parameters():
                if f'attentions.{session - 1}' in name or f'fc_auxs.{session - 1}' in name:
                    params += p.numel()
            # Add base encoder parameters
            params += sum(p.numel() for n, p in self.named_parameters()
                          if 'attentions' not in n and 'fc_auxs' not in n)
            return params

    def calculate_flops(self, input_size=(1, 3, 224, 224), session=None):
        """Calculate FLOPs for the entire model or specific session"""
        device = next(self.parameters()).device
        dummy_input = torch.randn(input_size).to(device)
        model_copy = deepcopy(self)

        if session is None:
            # Calculate FLOPs for entire model
            flops, _ = profile(model_copy, inputs=(dummy_input,), verbose=False)
            del model_copy
            return flops
        else:
            # For specific session, only calculate relevant modules
            session_flops = 0

            # Base encoder FLOPs
            encoder_copy = deepcopy(self.encoder)
            base_flops, _ = profile(encoder_copy, inputs=(dummy_input,), verbose=False)
            session_flops += base_flops
            del encoder_copy

            if session > 0:
                # Attention module FLOPs
                attention_copy = deepcopy(self.attentions[session - 1])
                feat_size = self.encoder(dummy_input).shape  # Get feature map size
                dummy_feat = torch.randn(feat_size).to(device)
                att_flops, _ = profile(attention_copy, inputs=(dummy_feat,), verbose=False)
                session_flops += att_flops
                del attention_copy

                # FC aux FLOPs
                fc_copy = deepcopy(self.fc_auxs[session - 1])
                dummy_fc_input = torch.randn(1, self.num_features).to(device)
                fc_flops, _ = profile(fc_copy, inputs=(dummy_fc_input,), verbose=False)
                session_flops += fc_flops
                del fc_copy

            return session_flops

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim//8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim//8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # Calculate the attention map
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width*height).permute(0,2,1)
        proj_key = self.key_conv(x).view(batch_size, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, width*height)

        # Apply the attention map to the input features
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)

        # Apply attention weights and add the original input
        out = self.gamma * out + x
        return out