修改1：将频域单元 FourierUnit 中写死的 BatchNorm2d 替换为可配置的 norm_layer（并在缺省时使用 GroupNorm），以与空域一致；
