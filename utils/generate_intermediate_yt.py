import torch as th

def generate_intermediate_yt(y0, gd_joints, mask, L, step):
    dir = gd_joints - y0
    dir = dir / th.norm(dir)
    yt = y0
    
    for t in range(step):
        if(th.norm(gd_joints - yt) >= L):
            yt = yt + L * dir
        else:
            if(th.norm(gd_joints - yt) <= 0.01):
                print(gd_joints.shape)
                return gd_joints
            yt = yt + th.norm(gd_joints - yt) * dir
    return yt


"""
def generate_intermediate_yt(y0, gt_joints, mask, L):
    dir = gt_joints - y0
    dir = dir / th.norm(dir)
    intermediate_values = []
    yt = y0

    while(th.norm(gt_joints - yt) >= L):
        yt = yt + L * dir
        intermediate_values.append(yt)

    intermediate_values.append(gt_joints)

    print("final : ",intermediate_values[2].shape)
    return intermediate_values
"""