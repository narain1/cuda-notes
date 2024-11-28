import torch, math

n_inp = 64
n_out = 64

d = 128
Q = torch.randn(n_out, d)
K = torch.randn(n_inp, d)
V = torch.randn(n_inp, d)
o = torch.randn(n_out, d)
l = torch.randn(n_out, l)

b_c = 16
b_r = 16
t_c = (n_inp + b_c - 1) // b_c
t_r = (n_out + b_r - 1) // b_r

scale_factor = 1 / math.sqrt(q.size(-1))

## q and o l split into t_r; k, v in t_c blocks
for i in range(t_r):
    q_i = q[i * b_r: (i + 1) * b_r]
    o_i = torch.zeros(b_r, d)
    l_i = torch.zeros(b_r, 1)
    m_i = torch.zeros((b_r, 1), -math.inf)
    last_m_i = m_i
    for j in range(t_c):
        k_j = k[j * b_c: (j+1) * b_c]
        v_j = v[j * b_c: (j+1) * b_c]
        s_i = scale_factor * (q_i @ k_j.T)
        m_i = torch.maximum(m_i, s_i.max(dim=-1, keepdim=True).values)
        p_i = torch.exp(s_i - m_i)
        l_i = torch.exp(last_m_i - m_i) * l_i + p_i.sum(dim=-1, keepdim=True)
        o_i = torch.exp(last_m_i - m_i) * o_i + p_i @ v_j
        last_m_i = m_i
    o_i = (1.0 / l_i) * o_i
    l_i = m_i + torch.log(l_i)
    o[i * b_r: (i+1)*b_r] = o_i
    l[i * b_r: (i+1)*b_r] = l_i
