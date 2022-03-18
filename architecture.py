from SwinIR.models.network_swinir import SwinIR as net

def IMDN(upscale=4):
    model = net(upscale=upscale, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=48, num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
    return model
