
class ExperimentFactory:
    

    def full_setup(rcount):
        mechs           = ['gp_add','nn','polynomial','sigmoid_add']
        noises          = [0.2,0.3,0.4]
        nodes           = [5,10,15]
        mask_probs      = [0.1, 0.3, 0.5]
        file_count      = 5
        pkl_name        = './res/exp_res_'+str(rcount)+'.pkl'
        
        return mechs,noises,nodes,mask_probs,file_count,pkl_name

    def toy_setup(rcount):
        mechs           = ['gp_add','polynomial','nn']#,'gp_add','polynomial','sigmoid_add']
        noises          = [0.3]
        nodes           = [10]#,20]#8,12,16]
        mask_probs      = [0.05,0.1,0.15,0.2,0.25]#.1,0.2,0.3]
        file_count      = 3
        pkl_name        = './res/toy_'+str(rcount)+'.pkl'
        
        return mechs,noises,nodes,mask_probs,file_count,pkl_name
    
    def comp_setup(rcount):
        mechs           = ['gp_add']#,'nn','polynomial','sigmoid_add']
        noises          = [0.3]
        nodes           = [5,10,15]
        mask_probs      = [0.05,0.15,0.25]
        file_count      = 5
        source_only     = [True,False]
        pkl_name        = './res/abl_res_'+str(rcount)+'.pkl'
        
        return mechs,noises,nodes,mask_probs,source_only,file_count,pkl_name

    def src_setup(rcount):
        mechs           = ['gp_add']#,'nn','polynomial','sigmoid_add']
        noises          = [0.3]
        nodes           = [5,10,15]
        mask_probs      = [0.05,0.15,0.25]
        file_count      = 5
        source_only     = [True,False]
        pkl_name        = './res/src_res_'+str(rcount)+'.pkl'
        
        return mechs,noises,nodes,mask_probs,source_only,file_count,pkl_name
    
    def src_setup_mnar(rcount):
        mechs           = ['gp_add']#,'nn','polynomial','sigmoid_add']
        noises          = [0.3]
        nodes           = [5,10,15]
        mask_probs      = [0.05,0.15,0.25]
        file_count      = 5
        source_only     = [False]
        pkl_name        = './res/src_res_mnar_'+str(rcount)+'.pkl'
        
        return mechs,noises,nodes,mask_probs,source_only,file_count,pkl_name

    
    def cd_setup(rcount):
        mechs           = ['gp_add']#,'nn','polynomial','sigmoid_add']
        noises          = [0.3]
        nodes           = [5,10,15]
        mask_probs      = [0.05,0.15,0.25]
        file_count      = 5
        source_only     = [True,False]
        pkl_name        = './res/cd_resmiss_'+str(rcount)+'.pkl'
        
        return mechs,noises,nodes,mask_probs,source_only,file_count,pkl_name
    
    def cd_mnar_setup(rcount):
        mechs           = ['gp_add']#,'nn','polynomial','sigmoid_add']
        noises          = [0.3]
        nodes           = [5,10,15]
        mask_probs      = [0.05,0.15,0.25]
        file_count      = 5
        source_only     = [True,False]
        pkl_name        = './res/cd_resmiss_mnar_'+str(rcount)+'.pkl'
        
        return mechs,noises,nodes,mask_probs,source_only,file_count,pkl_name

    def mnar_setup(rcount):
        mechs           = ['gp_add']#,'nn','polynomial','sigmoid_add']
        noises          = [0.3]
        nodes           = [5,10,15]
        mask_probs      = [0.05,0.15,0.25]
        file_count      = 5
        source_only     = [True,False]
        pkl_name        = './res/abl_res_mnar_'+str(rcount)+'.pkl'
        
        return mechs,noises,nodes,mask_probs,source_only,file_count,pkl_name
    
    def real_setup(rcount):
        mask_probs      = [0.05,0.1,0.15,0.2,0.25]
        pkl_name        = './res/exp_realworld.pkl'
        return mask_probs,pkl_name

    