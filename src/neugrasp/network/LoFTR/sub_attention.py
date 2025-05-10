import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)


class FeedForward(nn.Module):
    def __init__(self, dim, hid_dim, dp_rate):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, dim)
        self.dp = nn.Dropout(dp_rate)
        self.activ = nn.ReLU()

    def forward(self, x):
        x = self.dp(self.activ(self.fc1(x)))
        x = self.dp(self.fc2(x))
        return x


class Attention(nn.Module):
    def __init__(self, dim, dp_rate):
        super(Attention, self).__init__()
        self.q_fc = nn.Linear(dim, dim, bias=False)
        self.k_fc = nn.Linear(dim, dim, bias=False)
        self.v_fc = nn.Linear(dim, dim, bias=False)
        self.attn_fc = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim),
        )
        self.out_fc = nn.Linear(dim, dim)
        self.dp = nn.Dropout(dp_rate)

        self.apply(weights_init)

    def forward(self, q, k, pos, mask=None):
        q = q.permute(0, 2, 3, 1)
        k = k.permute(0, 2, 3, 1)

        q = self.q_fc(q)
        k = self.k_fc(k)
        v = self.v_fc(k)

        if pos is not None:
            attn = k - q + pos
        else:
            attn = k - q
        attn = self.attn_fc(attn)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = torch.sigmoid(attn)
        attn = self.dp(attn)

        if pos is not None:
            x = (v + pos) * attn
        else:
            x = v * attn

        x = self.dp(self.out_fc(x))
        return x.permute(0, 3, 1, 2)


class Attention_Conv(nn.Module):
    def __init__(self, dim, dp_rate):
        super().__init__()
        self.q_fc = nn.Conv2d(dim, dim, bias=False, kernel_size=1)
        self.k_fc = nn.Conv2d(dim, dim, bias=False, kernel_size=1)
        self.v_fc = nn.Conv2d(dim, dim, bias=False, kernel_size=1)
        self.attn_fc = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        )
        self.out_fc = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.dp = nn.Dropout(dp_rate)

        self.apply(weights_init)

    def forward(self, q, k, pos, mask=None):
        n, c, h, w = q.shape

        q = self.q_fc(q)
        k = self.k_fc(k)
        v = self.v_fc(k)

        if pos is not None:
            attn = k - q + pos
        else:
            attn = k - q
        attn = self.attn_fc(attn)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = F.softmax(attn.flatten(-2), dim=-1).reshape(n, c, h, w)
        attn = self.dp(attn)

        if pos is not None:
            x = (v + pos) * attn
        else:
            x = v * attn

        x = self.dp(self.out_fc(x))
        return x
