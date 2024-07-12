import numpy as np



def text_emb_pos(txt, tokenizer):
    emb_pos = []

    for i in range(1, len(txt)+1):
        et = tokenizer.encode(txt[:i])
        emb_pos.append(list(zip(et, list(range(len(et))))))

    return emb_pos

def create_mask(emb_pos):
    total_len = len(emb_pos)
    res = []
    mask = np.zeros([total_len, total_len])
    for embt, m in zip(emb_pos, mask):
        res.append(embt[-1])
        m[len(res)-1] = 1
        for ie in embt[:-1]:
            for j, je in enumerate(res[:-1]):
                if je == ie:
                    m[j] = 1
                    break
    return np.array(res), mask

def create_mask_embedding(txt, tokenizer):
    emb_pos = text_emb_pos(txt, tokenizer)
    res, mask = create_mask(emb_pos)    

    return res[:, 0], res[:, 1], mask

def sum_masks(o_mask_size, masks):
    iy = 0
    ix = 0
    o_mask = np.zeros((o_mask_size, o_mask_size))

    for m in masks:
        ms = len(m)
        o_mask[iy:iy+ms, ix:ix+ms] += m
        o_mask[iy+ms:, ix:ix+ms] = m[-1]
        iy+=ms
        ix+=ms
    return o_mask


def process_token(t, tokenizer):
    td = tokenizer.decode(t)
    if len(td)==1 or "�" in td:
        emb = np.array([t]) 
        pos = np.zeros((1))
        mask = np.ones((1,1))
    else:
        emb, pos, mask = create_mask_embedding(td, tokenizer)
        if len(emb) == 1:
            raise Exception("Used!")
        
    return emb, pos, mask


def encode_em(txt, tokenizer):
    tokens = tokenizer.encode(txt)

    res_emb = []
    res_pos = []
    partial_masks = []
    pos_len = 0

    for t in tokens:
        emb, pos, mask = process_token(t, tokenizer)

        res_emb.extend(emb)
        res_pos.extend(pos+pos_len)
        pos_len += pos[-1] + 1
        partial_masks.append(mask)

    return np.array(res_emb), np.array(res_pos), sum_masks(len(res_emb), partial_masks)