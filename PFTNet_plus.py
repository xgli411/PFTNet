from layers import *


from model import *
from archs.hat_arch import HAT
from archs.fftformer_arch import *


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8, ResBlock=ResBlock):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class DBlock(nn.Module):
    def __init__(self, channel, num_res=8, ResBlock=ResBlock):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class AFF(nn.Module):
    def __init__(self, in_channel, out_channel, BasicConv=BasicConv):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)

class SCM(nn.Module):
    def __init__(self, out_plane, BasicConv=BasicConv, inchannel=3):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(inchannel, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane-inchannel, kernel_size=1, stride=1, relu=True)
        )

        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)

class FAM(nn.Module):
    def __init__(self, channel, BasicConv=BasicConv):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out

class DeepRFT_plus(nn.Module):
    def __init__(self, num_res=20, inference=False):
        super(DeepRFT_plus, self).__init__()

        self.num_trans = 6
        self.num_blocks=[1, 2, 3, 4]
        self.ffn_expansion_factor=3

        self.inference = inference
        if not inference:
            BasicConv = BasicConv_do
            ResBlock = ResBlock_do_fft_bench
        else:
            BasicConv = BasicConv_do_eval
            ResBlock = ResBlock_do_fft_bench_eval

        base_channel = 32

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res, ResBlock=ResBlock),
            EBlock(base_channel*2, num_res, ResBlock=ResBlock),
            EBlock(base_channel*4, num_res, ResBlock=ResBlock),
        ])
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=base_channel, ffn_expansion_factor=self.ffn_expansion_factor, bias=False) for i in
            range(self.num_blocks[0])])

        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(base_channel * 2 ** 1), ffn_expansion_factor=self.ffn_expansion_factor,
                             bias=False) for i in range(self.num_blocks[1])])
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(base_channel * 2 ** 2), ffn_expansion_factor=self.ffn_expansion_factor,
                             bias=False) for i in range(self.num_blocks[3])])


        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res, ResBlock=ResBlock),
            DBlock(base_channel * 2, num_res, ResBlock=ResBlock),
            DBlock(base_channel, num_res, ResBlock=ResBlock)
        ])

        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(base_channel * 2 ** 2), ffn_expansion_factor=self.ffn_expansion_factor,
                             bias=False, att=True) for i in range(self.num_blocks[2])])
        #self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(base_channel * 2 ** 1), ffn_expansion_factor=self.ffn_expansion_factor,
                             bias=False, att=True) for i in range(self.num_blocks[0])])

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(base_channel), ffn_expansion_factor=self.ffn_expansion_factor,
                             bias=False, att=True) for i in range(self.num_blocks[0])])
        
        
        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel*1, BasicConv=BasicConv),
            AFF(base_channel * 7, base_channel*2, BasicConv=BasicConv)
        ])


        # 
        self.prompt2 = PromptGenBlock(prompt_dim=32,prompt_len=5,prompt_size = 64,lin_dim = base_channel*2)
        self.conv2 = BasicConv(32,32,kernel_size=3,relu=True, stride=1)

        self.decoder_level2_2 = nn.Sequential(*[
            TransformerBlock(dim=int(base_channel * 2 ** 1+32), ffn_expansion_factor=self.ffn_expansion_factor,
                             bias=False, att=True) for i in range(self.num_blocks[0])])
        self.reduce_noise_level2 = nn.Conv2d(96,base_channel*2,kernel_size=1,bias=False)

        self.prompt3 = PromptGenBlock(prompt_dim=base_channel * 2 ,prompt_len=5,prompt_size = 64,lin_dim = base_channel * 2 ** 2)
        self.conv3 = BasicConv(base_channel * 2,base_channel * 2,kernel_size=3,relu=True, stride=1)
        self.decoder_level3_3 = nn.Sequential(*[
            TransformerBlock(dim=int(base_channel * 6), ffn_expansion_factor=self.ffn_expansion_factor,
                             bias=False, att=True) for i in range(self.num_blocks[0])])
        self.reduce_noise_level3 = nn.Conv2d(base_channel * 6,base_channel * 2 ** 2,kernel_size=1,bias=False)



        self.FAM1 = FAM(base_channel * 4, BasicConv=BasicConv)
        self.SCM1 = SCM(base_channel * 4, BasicConv=BasicConv)
        self.FAM2 = FAM(base_channel * 2, BasicConv=BasicConv)
        self.SCM2 = SCM(base_channel * 2, BasicConv=BasicConv)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)
        res1 = self.encoder_level1(res1)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)
        res2 = self.encoder_level2(res2)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)
        z = self.encoder_level3(z)

        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)

        z = self.decoder_level3(z)
        z = self.Decoder[0](z)
        z_param = self.prompt3(z)
        z_param = self.conv3(z_param)
        z = torch.cat([z, z_param], 1)
        z = self.decoder_level3_3(z)
        z = self.reduce_noise_level3(z)
        z_ = self.ConvsOut[0](z)


        z = self.feat_extract[3](z)
        if not self.inference:
            outputs.append(z_+x_4)
        


        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        #print('z',z.shape)
        z = self.decoder_level2(z)
        z = self.Decoder[1](z)
        z_param = self.prompt2(z)
        z_param = self.conv2(z_param)
        z = torch.cat([z, z_param], 1)
        z = self.decoder_level2_2(z)
        z = self.reduce_noise_level2(z)
        z_ = self.ConvsOut[1](z)


        z = self.feat_extract[4](z)
        if not self.inference:
            outputs.append(z_+x_2)
    

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.decoder_level1(z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        if not self.inference:
            outputs.append(z + x)
            return outputs[::-1]
        else:
            return z + x
class DeepRFT_plus_flops(nn.Module):
    def __init__(self, num_res=20, inference=True):
        super(DeepRFT_plus_flops, self).__init__()

        self.num_trans = 6
        self.num_blocks=[1, 2, 3, 4]
        self.ffn_expansion_factor=3

        self.inference = inference
        
        ResBlock = ResBlock_fft_bench

        base_channel = 32

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res, ResBlock=ResBlock),
            EBlock(base_channel*2, num_res, ResBlock=ResBlock),
            EBlock(base_channel*4, num_res, ResBlock=ResBlock),
        ])
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=base_channel, ffn_expansion_factor=self.ffn_expansion_factor, bias=False) for i in
            range(self.num_blocks[0])])

        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(base_channel * 2 ** 1), ffn_expansion_factor=self.ffn_expansion_factor,
                             bias=False) for i in range(self.num_blocks[1])])
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(base_channel * 2 ** 2), ffn_expansion_factor=self.ffn_expansion_factor,
                             bias=False) for i in range(self.num_blocks[3])])


        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res, ResBlock=ResBlock),
            DBlock(base_channel * 2, num_res, ResBlock=ResBlock),
            DBlock(base_channel, num_res, ResBlock=ResBlock)
        ])

        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(base_channel * 2 ** 2), ffn_expansion_factor=self.ffn_expansion_factor,
                             bias=False, att=True) for i in range(self.num_blocks[2])])
        #self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(base_channel * 2 ** 1), ffn_expansion_factor=self.ffn_expansion_factor,
                             bias=False, att=True) for i in range(self.num_blocks[0])])

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(base_channel), ffn_expansion_factor=self.ffn_expansion_factor,
                             bias=False, att=True) for i in range(self.num_blocks[0])])


        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel*1, BasicConv=BasicConv),
            AFF(base_channel * 7, base_channel*2, BasicConv=BasicConv)
        ])

        self.window_blocks = nn.ModuleList([
            SwinTransformerBlock(dim=base_channel * 4,
                                 input_resolution=(64, 64),
                                 num_heads=8, window_size=4,
                                 shift_size=0 if (i % 2 == 0) else 2,
                                 mlp_ratio=4,
                                 qkv_bias=True,
                                 drop=0., attn_drop=0.)
            for i in range(self.num_trans)])
        # 
        self.prompt2 = PromptGenBlock(prompt_dim=32,prompt_len=5,prompt_size = 64,lin_dim = base_channel*2)
        self.conv2 = BasicConv(32,32,kernel_size=3,relu=True, stride=1)

        self.decoder_level2_2 = nn.Sequential(*[
            TransformerBlock(dim=int(base_channel * 2 ** 1+32), ffn_expansion_factor=self.ffn_expansion_factor,
                             bias=False, att=True) for i in range(self.num_blocks[0])])
        self.reduce_noise_level2 = nn.Conv2d(96,base_channel*2,kernel_size=1,bias=False)

        self.prompt3 = PromptGenBlock(prompt_dim=base_channel * 2 ,prompt_len=5,prompt_size = 64,lin_dim = base_channel * 2 ** 2)
        self.conv3 = BasicConv(base_channel * 2,base_channel * 2,kernel_size=3,relu=True, stride=1)
        self.decoder_level3_3 = nn.Sequential(*[
            TransformerBlock(dim=int(base_channel * 6), ffn_expansion_factor=self.ffn_expansion_factor,
                             bias=False, att=True) for i in range(self.num_blocks[0])])
        self.reduce_noise_level3 = nn.Conv2d(base_channel * 6,base_channel * 2 ** 2,kernel_size=1,bias=False)



        self.FAM1 = FAM(base_channel * 4, BasicConv=BasicConv)
        self.SCM1 = SCM(base_channel * 4, BasicConv=BasicConv)
        self.FAM2 = FAM(base_channel * 2, BasicConv=BasicConv)
        self.SCM2 = SCM(base_channel * 2, BasicConv=BasicConv)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)
        res1 = self.encoder_level1(res1)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)
        res2 = self.encoder_level2(res2)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)
        z = self.encoder_level3(z)

        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)

        z = self.decoder_level3(z)
        z = self.Decoder[0](z)
        z_param = self.prompt3(z)
        z_param = self.conv3(z_param)
        z = torch.cat([z, z_param], 1)
        z = self.decoder_level3_3(z)
        z = self.reduce_noise_level3(z)
        z_ = self.ConvsOut[0](z)


        z = self.feat_extract[3](z)
        if not self.inference:
            outputs.append(z_+x_4)



        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        #print('z',z.shape)
        z = self.decoder_level2(z)
        z = self.Decoder[1](z)
        z_param = self.prompt2(z)
        z_param = self.conv2(z_param)
        z = torch.cat([z, z_param], 1)
        z = self.decoder_level2_2(z)
        z = self.reduce_noise_level2(z)
        z_ = self.ConvsOut[1](z)


        z = self.feat_extract[4](z)
        if not self.inference:
            outputs.append(z_+x_2)


        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.decoder_level1(z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        if not self.inference:
            outputs.append(z + x)
            return outputs[::-1]
        else:
            return z + x
if __name__ == '__main__':
    torch.manual_seed(100)
    x = torch.rand((1, 3, 256, 256))
    # x = torch.rand((1, 3, 1280, 720))
    model = DeepRFT_flops()
    # model = MIMOUNetPlus()
    y = model(x)
    print(y.shape)


