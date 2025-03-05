from dataset.dataset_decouple_SR import *
       
def gen_data_loader(opt, random_selection=False, crop_flag=True, flip_flag=True):  
    train_dir_LR=opt['train_dir_LR']
    train_dir_HR=opt['train_dir_HR']
    test_dir_LR=opt['test_dir_LR']
    test_dir_HR=opt['test_dir_HR']
    GT_tag_list=opt['category']
    size=opt['size']
    batch_size=opt['train']['batch_size']
    device=opt['device']
    num_train=opt['num_train']
    num_test=opt['num_test']
    factor_list=opt['factor_list']    
    up_factor = opt['up_factor']
    noise_level = opt['noise_level']
    output_list = opt['output_list']
    denoise = opt['denoise']
    w0 = opt['degeneration_w0']
    read_LR = opt['read_LR']
    # generate directions for different components
    train_dir_GT_HR_list = []
    test_dir_GT_HR_list = []
    train_dir_GT_LR_list = []
    test_dir_GT_LR_list = []
    for i in range(len(GT_tag_list)):
        train_dir_GT_HR_list.append(os.path.join(train_dir_HR, GT_tag_list[i]))
        test_dir_GT_HR_list.append(os.path.join(test_dir_HR, GT_tag_list[i]))
        train_dir_GT_LR_list.append(os.path.join(train_dir_LR, GT_tag_list[i]))
        test_dir_GT_LR_list.append(os.path.join(test_dir_LR, GT_tag_list[i]))
    #if denoise == None:
    train_dataset = Dataset_decouple_SR(
        GT_dir_list_DS=train_dir_GT_HR_list, GT_dir_list_D=train_dir_GT_LR_list,
        size=size, device=device, noise_level=noise_level, output_list=output_list, denoise=denoise, 
        train_flag=True, num_file=num_train, up_factor=up_factor, factor_list=factor_list, 
        random_selection=True, crop_flag=True, flip_flag=True, w0=w0, read_LR=read_LR
    )
    val_dataset = Dataset_decouple_SR(
        GT_dir_list_DS=test_dir_GT_HR_list, GT_dir_list_D=test_dir_GT_LR_list,
        size=size, device=device, noise_level=noise_level, output_list=output_list, denoise=denoise, 
        train_flag=False, num_file=num_test, up_factor=up_factor, factor_list=factor_list, 
        random_selection=False, crop_flag=False, flip_flag=False, w0=w0, read_LR=read_LR
    )
    '''else:
        train_dataset = Dataset_denoise(
            GT_dir_list_D=train_dir_GT_LR_list, device=device, num_file=num_train, up_factor=up_factor, factor_list=factor_list,  size=size, 
            train_flag=True, noise_level=noise_level, random_selection=random_selection, crop_flag=crop_flag, flip_flag=flip_flag
        )
        val_dataset = Dataset_denoise(
            GT_dir_list_D=test_dir_GT_LR_list, device=device, num_file=num_test, up_factor=up_factor, factor_list=factor_list, size=size, 
            train_flag=False, noise_level=noise_level, random_selection=False, crop_flag=False, flip_flag=False
        )'''
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=1)
    num_train_image = len(train_dataset)
    # return dataloader and the number of enumerations
    return train_loader, val_loader, num_train_image


if __name__ == '__main__':
    #img_1 = np.array(Image.open(r"D:\CQL\codes\microscopy_decouple\data\STED_data_raw\Composed\checked\2_confocal.tiff"))    
    #img_2 = np.array(Image.open(r"D:\CQL\codes\microscopy_decouple\visualization\synthesized\1.tiff"))
    #print(np.mean(img_1), np.mean(img_2))
    if 1:
        train_dir_LR = r'D:\CQL\codes\microscopy_decouple\data\train_LR'
        test_dir_LR = r'D:\CQL\codes\microscopy_decouple\data\test_LR'
        train_dir_HR = r'D:\CQL\codes\microscopy_decouple\data\train_HR'
        test_dir_HR = r'D:\CQL\codes\microscopy_decouple\data\test_HR'

        size = 512
        lr = 6e-5
        epoches = 2000
        epoches_per_test = 5
        epoches_per_save = 20
        name = '12.2_multitask'
        batch_size = 1
        Input_tag_list = ['multiorganelle_SIM']
        #Input_HRLR_list = ['LR']
        GT_tag_list = ['Microtubes', 'Mito', 'Ly']
        factor_list = [1, 1, 1]
        device = 'cuda'
        num_train = 20
        num_test = 10
        RL = None
        train_loader, test_loader, num_train_image, mean, std = gen_data_loader(train_dir_LR=train_dir_LR, train_dir_HR=train_dir_HR, test_dir_LR=test_dir_LR, test_dir_HR=test_dir_HR, 
                    size=size, batch_size=batch_size, device=device, num_train=num_train, num_test=num_test, RL=RL, 
                    GT_tag_list=GT_tag_list, factor_list=factor_list)
        import cv2
        save_dir_mito = r'D:\CQL\codes\microscopy_decouple\visualization\mito'
        save_dir_microtubes = r"D:\CQL\codes\microscopy_decouple\visualization\microtubes"
        save_dir_synthesized = r'D:\CQL\codes\microscopy_decouple\visualization\synthesized'
        check_existence(save_dir_synthesized)
        reduction = 1
        std = 50
        for batch_index, (Input, GT_DS, GT_D, GT_S) in enumerate(train_loader):
            print(Input.size(), GT_DS.size(), GT_D.size(), GT_S.size())
            '''try:                     
                #print(torch.max(Input))
                Input = np.uint16(to_cpu(Input.squeeze(0).permute(1,2,0)) * std / reduction)                
                GT_D = np.uint16(to_cpu(GT_D.squeeze(0).permute(1,2,0)) * std / reduction)   
                #print(np.max(Input))
                save_dir_file = os.path.join(save_dir_synthesized, '{}_syn.tiff'.format(batch_index+1))
                cv2.imencode('.tiff', np.uint16(Input))[1].tofile(save_dir_file)            
                save_dir_file = os.path.join(save_dir_synthesized, '{}_mito.tiff'.format(batch_index+1))
                cv2.imencode('.tiff', np.uint16(GT_D[:,:,1]))[1].tofile(save_dir_file)  
                save_dir_file = os.path.join(save_dir_synthesized, '{}_microtubes.tiff'.format(batch_index+1))
                cv2.imencode('.tiff', np.uint16(GT_D[:,:,0]))[1].tofile(save_dir_file)                      
                save_dir_file = os.path.join(save_dir_synthesized, '{}_Ly.tiff'.format(batch_index+1))
                cv2.imencode('.tiff', np.uint16(GT_D[:,:,2]))[1].tofile(save_dir_file)                                      
            except:
                pass'''