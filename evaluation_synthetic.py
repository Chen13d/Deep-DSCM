import os, sys
from tqdm import tqdm

from options.options import parse
from dataset.read_prepared_data import *

cwd = os.getcwd()
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(parent_dir)

from loss.SSIM_loss import SSIM
from loss.NRMAE import nrmae
from loss.FRC_cal import estimate_resolution_via_fft

from net.make_model import *
from dataset.gen_datasets import *


#options of .yml format in "options" folder
opt_path = 'options/Synthetic_eval.yml'

# read options
opt = parse(opt_path=os.path.join(cwd, opt_path))
# set rank of GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_rank']

import torch
if torch.cuda.is_available():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('-----------------------------Using GPU-----------------------------')
else:
    print('-----------------------------Using CPU-----------------------------')




def main():
    toolbox = ToolBox(opt=opt)    
    net_main = DSCM_with_dataset(opt, in_channels=1, num_classes=len(opt['category']), model_name_G=opt['net_G']['model_decouple_name'], 
                        model_name_D=opt['net_D']['model_name'], initialize=opt['net_G']['initialize'], mode=opt['net_G']['mode_decouple'], 
                        scheduler_name=opt['train']['scheduler'], device=device, weight_list=opt['net_G']['weight_decouple'], lr_G=opt['train']['lr_G'], lr_D=opt['train']['lr_D'])
    net_main.net_G = torch.load(opt['net_G']['pretrain_dir'], weights_only=False)
    #net_main.net_G.train()
    # "old" = read data pairs, "new" = generate pseudo data pairs
    if opt['read_version'] == "real-time":
        val_loader = gen_degradation_dataloader(
            GT_tag_list=opt['category'], 
            noise_level=opt['noise_level'], 
            w0_T=6.9, 
            factor_list=opt['factor_list'], 
            STED_resolution_dict=opt['resolution'], 
            target_resolution=opt['degradation_resolution'], 
            generate_FLIM=opt['FLIM'], 
            degradation_method=opt['degradation_method'], 
            average=opt['average'], 
            read_LR=opt['read_LR'], 
            num_file_train=opt['num_file_train'], 
            num_file_val=opt['num_file_val'], 
            size=opt['size'], 
            num_workers=opt['num_workers'], 
            cwd=cwd, 
            device=device, 
            real_time=True, 
            eval_flag=True
        )
        num_val_image = opt['num_file_val']
    elif opt['read_version'] == "prepared":
        combination_name = "_".join(opt['category']) + f"_{opt['degradation_resolution']}" + f"_{opt['noise_level']}" + f"_{opt['average']}"
        read_dir_train = os.path.join(r'data\prepared_data\train', combination_name)
        read_dir_val = os.path.join(r'data\prepared_data\val', combination_name)
        #train_loader, val_loader, num_tr  ain_image = gen_prepared_dataloader(read_dir_train=read_dir_train, read_dir_val=read_dir_val, num_file_train=opt['num_train'], 
        #                                                        num_file_val=opt['num_test'], num_org=len(opt['category']), org_list=opt['category'], size=opt['size'], 
        #                                                        batch_size=opt['train']['batch_size'], device=opt['device'])
        train_dataset = prepared_dataset(read_dir=read_dir_train, num_file=opt['num_file_train'], num_org=len(opt['category']), org_list=opt['category'], size=opt['size'], device=opt['device'])
        val_dataset = prepared_dataset(read_dir=read_dir_val, num_file=opt['num_file_val'], num_org=len(opt['category']), org_list=opt['category'], size=opt['size'], device=opt['device'])
        if opt['num_workers']:
            train_loader = DataLoader(train_dataset, shuffle=True, batch_size=opt['train']['batch_size'], num_workers=opt['num_workers'], persistent_workers=True, pin_memory=True)
        else:
            train_loader = DataLoader(train_dataset, shuffle=True, batch_size=opt['train']['batch_size'])
        val_loader = DataLoader(val_dataset, shuffle=False, batch_size=1)
        num_train_image = opt['num_file_train']
    # generate folders for validation
    toolbox.make_folders()
    Input_list = []
    GT_list = []
    GT_D_list = []
    sta_list = []
    # list for validation loss
    mae_list = []
    ssim_list = []
    SSIM_criterion = SSIM().to(device)
    pearson_coef_list = []
    print('======================== evaluating ========================')    
    bar = tqdm(total=num_val_image)
    # enumerate in test Dataloader 
    with torch.no_grad():
        for batch_index, data in enumerate(val_loader):
            Input, GT_DS, GT_D, _, _, _, sta = data
            fake_main = net_main.feed_data(Input=Input, GT=GT_DS)
            loss_main, pearson_coef = net_main.validation(mask=None)
            Input = Input / (torch.max(Input) - torch.min(Input))
            fake_main = fake_main / (torch.max(fake_main) - torch.min(fake_main))
            GT_DS = GT_DS / (torch.max(GT_DS) - torch.min(GT_DS))
            # append to list for epoches to save
            Input_list.append(Input)
            GT_list.append([fake_main, GT_DS])
            sta_list.append(sta)
            GT_D_list.append(to_cpu((GT_D*sta["Input_std"]).squeeze(0).permute(1,2,0)))
            # cal NRMAE
            mae_loss = nrmae(fake_main, GT_DS) 
            # cal SSIM
            for i in range(fake_main.size()[1]):
                temp_fake = fake_main[:,i:i+1,:,:].detach()
                temp_GT = GT_DS[:,i:i+1,:,:].detach()
                temp_fake = temp_fake / torch.max(temp_fake)
                temp_GT = temp_GT / torch.max(temp_GT)
                SSIM_value = SSIM_criterion(temp_fake, temp_GT)
            mae_list.append(mae_loss.item())
            ssim_list.append(SSIM_value.item())
            # PCC loss
            pearson_coef_list.append(pearson_coef.item())
            bar.update(1)
        pearson_aver = np.mean(pearson_coef_list)
        pearson_coef_list.append(pearson_aver)
        # save val stack and model
        toolbox.gen_validation_images_with_dataset(data_list=[Input_list, GT_list, sta_list])
        toolbox.save_val_list(name="main")

        save_dir = r"C:\Users\18923\OneDrive\Work\DSRM_paper\synthetic_data_eval\Real_comparison\data"
        check_existence(save_dir)
        save_list = toolbox.val_list
        size = opt['size']
        for index in range(len(save_list)):
            temp_img = save_list[index]
            Input = temp_img[:size, :size]
            for org_index in range(len(opt['category'])):
                GT = temp_img[:size, (org_index+1)*size:(org_index+2)*size]
                pred = temp_img[size:2*size, (org_index+1)*size:(org_index+2)*size]
                tifffile.imwrite(os.path.join(save_dir, f"{index}_GT_{org_index}.tif"), np.uint8(GT))
                tifffile.imwrite(os.path.join(save_dir, f"{index}_pred_{org_index}.tif"), np.uint8(pred))
                # save GT_D for RSP RSE
                temp_GT_D = GT_D_list[index][:,:,org_index]
                tifffile.imwrite(os.path.join(save_dir, f"{index}_GT_D_{org_index}.tif"), np.uint16(temp_GT_D))

            tifffile.imwrite(os.path.join(save_dir, f"{index}_Input.tif"), Input)
        

    bar.close()

    print(np.mean(mae_list))
    print(np.mean(ssim_list))
    print(np.mean(pearson_coef_list))

    top3_with_index = sorted(enumerate(ssim_list), key=lambda x: x[1], reverse=True)[:20]
    # 提取值和索引
    values = [v for i, v in top3_with_index]
    indices = [i for i, v in top3_with_index]
    print(values)   # [89, 45, 23]
    print(indices)  # [4, 2, 3]

    return mae_list, ssim_list, pearson_coef_list


main()




if 0:
    import os, csv, numpy as np, pandas as pd

    # ------------------------------------------------------------
    # 1. 组合、文件名
    # ------------------------------------------------------------
    binary_paths = [
        [0,   0, 0, 0], [0,   0, 0, 1], [0,   0, 1, 0], [0,   0, 1, 1],
        [0,   1, 0, 0], [0,   1, 0, 1], [0,   1, 1, 0], [0,   1, 1, 1],
        [0.1, 0, 0, 0], [0.1, 0, 0, 1], [0.1, 0, 1, 0], [0.1, 0, 1, 1],
        [0.1, 1, 0, 0], [0.1, 1, 0, 1], [0.1, 1, 1, 0], [0.1, 1, 1, 1],
    ]

    base   = r"D:\CQL\codes\microscopy_decouple\Validation\DSCM_NPCs_Mito_inner_Membrane_228_0_1_DSCM_384_Unet"
    suffix = r"_real-time_1000_epoches\weights\1\main_G.pth"

    file_names   = []
    group_names  = []           # <—— 存别名，用于最终列头

    # 组装完整路径 & 别名
    for b0, b1, b2, b3 in binary_paths:
        file_names.append(
            f"{base}_fea_loss_{b0}_SSIM_loss_{b1}_grad_loss_{b2}_GAN_loss_{b3}{suffix}"
        )

        # -------- 生成别名 --------
        name = "MSE"
        if b0 > 0:   name += "+Fea"
        if b1 == 1:  name += "+SSIM"
        if b2 == 1:  name += "+Grad"
        if b3 == 1:  name += "+GAN"
        group_names.append(name)

    # ------------------------------------------------------------
    # 2. 跑实验，收集 raw_rows
    # ------------------------------------------------------------
    raw_rows, summary_rows = [], []

    for (b0, b1, b2, b3), fname, gname in zip(binary_paths, file_names, group_names):
        print(fname)
        opt["net_G"]["pretrain_dir"] = fname

        MAE_out, SSIM_out, PCC_out = main()          # 列表或 float
        # 保证都是 float 数组，去掉可能的引号
        def clean(arr): return np.asarray([float(str(v).lstrip("'")) for v in np.atleast_1d(arr)])

        MAE_arr, SSIM_arr, PCC_arr = map(clean, [MAE_out, SSIM_out, PCC_out])

        summary_rows.append({
            "fea_loss": b0, "SSIM_loss": b1, "grad_loss": b2, "GAN_loss": b3,
            "config": fname, "group": gname,
            "MAE_mean": MAE_arr.mean(), "SSIM_mean": SSIM_arr.mean(), "PCC_mean": PCC_arr.mean(),
        })

        for idx, (mae, ssim, pcc) in enumerate(zip(MAE_arr, SSIM_arr, PCC_arr)):
            raw_rows.append({
                "img_idx": idx,
                "config":  fname,      # 用于 pivot
                "MAE": mae, "SSIM": ssim, "PCC": pcc,
            })

    # ------------------------------------------------------------
    # 3. 构建 MultiIndex 列（config→metric），再把最外层换成 group_name
    # ------------------------------------------------------------
    df_raw = pd.DataFrame(raw_rows)

    # pivot 到 (img_idx × config)：
    mae_tbl  = df_raw.pivot(index="img_idx", columns="config", values="MAE")
    ssim_tbl = df_raw.pivot(index="img_idx", columns="config", values="SSIM")
    pcc_tbl  = df_raw.pivot(index="img_idx", columns="config", values="PCC")

    # 交错拼接：MAE→SSIM→PCC
    pieces = []
    for cfg in file_names:
        pieces.extend([mae_tbl[cfg], ssim_tbl[cfg], pcc_tbl[cfg]])
    df_wide = pd.concat(pieces, axis=1)

    # MultiIndex 列：外层先用 file_names，占位
    level0 = np.repeat(file_names, 3)
    level1 = ["MAE", "SSIM", "PCC"] * len(file_names)
    df_wide.columns = pd.MultiIndex.from_arrays([level0, level1])

    # -------- 把外层路径替换成 group_names --------
    path2group = dict(zip(file_names, group_names))
    df_wide.columns = pd.MultiIndex.from_arrays(
        [[path2group[p] for p in level0], level1]
    )

    # ------------------------------------------------------------
    # 4. 保存
    # ------------------------------------------------------------
    save_dir = r"C:\Users\18923\Desktop\DSRM_paper_on_submission_material\DSRM paper\loss_experiment\csvs"
    os.makedirs(save_dir, exist_ok=True)

    df_wide.to_csv(
        os.path.join(save_dir, "raw.csv"),
        index=True,
        float_format="%.18g",
        quoting=csv.QUOTE_NONE,
    )

    print("✔ raw.csv 生成完毕（列头 = 分组别名 / MAE·SSIM·PCC）")




if 0:
    import os, numpy as np, pandas as pd

    # ------------------------------------------------------------------
    # 1. 组合、文件名
    # ------------------------------------------------------------------
    binary_paths = [
        [0,   0, 0, 0], [0,   0, 0, 1], [0,   0, 1, 0], [0,   0, 1, 1],
        [0,   1, 0, 0], [0,   1, 0, 1], [0,   1, 1, 0], [0,   1, 1, 1],
        [0.1, 0, 0, 0], [0.1, 0, 0, 1], [0.1, 0, 1, 0], [0.1, 0, 1, 1],
        [0.1, 1, 0, 0], [0.1, 1, 0, 1], [0.1, 1, 1, 0], [0.1, 1, 1, 1],
    ]
    base = "D:\\CQL\\codes\\microscopy_decouple\\validation\\DSCM_NPCs_Mito_inner_Membrane_228_0_1_DSCM_384_Unet"
    suffix = "_real-time_1000_epoches\weights\\1\\main_G.pth"
    file_names = [
        f"{base}_fea_loss_{b0}_SSIM_loss_{b1}_grad_loss_{b2}_GAN_loss_{b3}{suffix}"
        for b0, b1, b2, b3 in binary_paths
    ]

    summary_rows, raw_rows = [], []

    for (b0, b1, b2, b3), fname in zip(binary_paths, file_names):
        print(fname)
        opt["net_G"]["pretrain_dir"] = fname
        MAE_out, SSIM_out, PCC_out = main()          # ← 列表或 float

        # 统一转 ndarray
        MAE_arr  = np.atleast_1d(MAE_out).astype(float)
        SSIM_arr = np.atleast_1d(SSIM_out).astype(float)
        PCC_arr  = np.atleast_1d(PCC_out).astype(float)

        # —— ① summary 行
        summary_rows.append({
            "fea_loss": b0, "SSIM_loss": b1, "grad_loss": b2, "GAN_loss":  b3,
            "config":   fname,
            "MAE_mean": MAE_arr.mean(),  "SSIM_mean": SSIM_arr.mean(),  "PCC_mean": PCC_arr.mean(),
        })

        # —— ② raw 行（每张图像一行）
        for idx, (mae, ssim, pcc) in enumerate(zip(MAE_arr, SSIM_arr, PCC_arr)):
            raw_rows.append({
                "img_idx": idx,
                "config":  fname,
                "MAE": mae,
                "SSIM": ssim,
                "PCC": pcc,
            })

    # 2. 生成 raw.csv —— 第一层列是 config，第二层列是 MAE/SSIM/PCC
    # ------------------------------------------------------------------
    df_raw = pd.DataFrame(raw_rows)          # raw_rows 为循环里收集的原始行
    # ① 先把三个指标各自 pivot 成 (img_idx × config) 形状
    mae_tbl  = df_raw.pivot(index="img_idx", columns="config", values="MAE")
    ssim_tbl = df_raw.pivot(index="img_idx", columns="config", values="SSIM")
    pcc_tbl  = df_raw.pivot(index="img_idx", columns="config", values="PCC")

    # ② 组装成 MultiIndex 列：
    #    外层：config，内层：metric，且顺序保证 MAE→SSIM→PCC
    pieces = []
    for cfg in file_names:                   # file_names = config 顺序列表
        pieces.append(mae_tbl[cfg])
        pieces.append(ssim_tbl[cfg])
        pieces.append(pcc_tbl[cfg])

    df_wide = pd.concat(pieces, axis=1)
    # 重建列索引 —— level 0 为 cfg，level 1 为 metric
    level0 = np.repeat(file_names, 3)                 # AAA,AAA,AAA,BBB,BBB,BBB,…
    level1 = ["MAE", "SSIM", "PCC"] * len(file_names) # MAE,SSIM,PCC,MAE,SSIM,PCC…
    df_wide.columns = pd.MultiIndex.from_arrays([level0, level1])

    # ③ 保存；pandas 会写两行 header
    save_dir = r"C:\Users\18923\Desktop\DSRM_paper_on_submission_material\DSRM paper\loss_experiment\csvs"
    os.makedirs(save_dir, exist_ok=True)
    df_wide.to_csv(os.path.join(save_dir, "raw.csv"), index=True)

    print("✔ raw.csv 生成完毕（第一行是 config，第二行是 MAE/SSIM/PCC）")

