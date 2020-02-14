stream_mode = "trapezoidal" # or variable, decrease sparsity when variable
cross_validation_folds = 25
random_seed = 1 #variable 100
olvf_C = 0.01 #variable 0.1
olvf_C2 = 0.00000001 #variable 0.1


olvf_option = 1 # 0, 1 or 2
olsf_C = 1
olsf_Lambda = 30
olsf_B = 1
olvf_B = 0.64
olsf_option = 2 # 0, 1 or 2

    
#def setParametersSpambase():
    #stream_mode = "trapezoidal" # or variable, decrease sparsity when variable
    #cross_validation_folds = 20
    #random_seed = 100 #variable 150
    #olvf_C = 0.01 #variable 0.01
    #olvf_Lambda = 30
    #olvf_B = 0.64
    #olvf_option = 1 # 0, 1 or 2
    #olsf_C = 1
    #olsf_Lambda = 30
    #olsf_B = 0.64
    #olsf_option = 1 # 0, 1 or 2
    #lr = 1
    #l2 = 0.1#1
    #eps =0

#def setParametersGerman():
    #stream_mode = "trapezoidal" # or variable, decrease sparsity when variable
    #cross_validation_folds = 20
    #random_seed = 100
    #olvf_C = 0.01
    #olvf_C2 = 0.001
    #olvf_B = 0.64
    #olsf_C = 0.1
    #olsf_Lambda = 30
    #olsf_B = 0.64
    #olsf_option = 1 # 0, 1 or 2

#def setParametersWDBC():
    #stream_mode = "trapezoidal" # or variable, decrease sparsity when variable
    #cross_validation_folds = 20
    #random_seed = 100
    #olvf_C = 0.00001
    #olvf_C2 = 1
    #olvf_option = 1 # 0, 1 or 2
    #olsf_C = 1
    #olsf_Lambda = 30
    #olsf_B = 0.64
    #olvf_B = 0.64
    #olsf_option = 2 # 0, 1 or 2 
    #lr = 0.1
    #l2 = 0
    #eps =0.01

#def setParametersWBC():
    #stream_mode = "trapezoidal" # or variable, decrease sparsity when variable
    #cross_validation_folds = 25
    #random_seed = 1 #variable 100
    #olvf_C = 0.001
    #olvf_C2 = 0.01 #variable 10
    #olvf_B = 0.64
    #olvf_option = 1 # 0, 1 or 2
    #olsf_C = 0.01
    #olsf_Lambda = 30
    #olsf_B = 1
    #olsf_option = 2 # 0, 1 or 2 # 0, 1 or 2 #sqrt of inconfi
    #lr = 0.1
    #l2 = 0.01#1
    #eps =0
    
#def setParametersMagic():
#    stream_mode = "trapezoidal" # or variable, decrease sparsity when variable
#    cross_validation_folds = 20 #variable 20
#    random_seed = 100
#    olvf_C = 0.75 #varible 1
#    olvf_Lambda = 30
#    olvf_B = 0.64
#    olvf_option = 1 # 0, 1 or 2
#    olsf_C = 0.1
#    olsf_Lambda = 30
#    olsf_B = 0.64
#    olsf_option = 1 # 0, 1 or 2
#    confnet_lr = 0.0001
#    confnet_l2 = 0.01
    
#def setParametersA8A():
    #phi = 0.01
    #cross_validation_folds = 2 #5
    #random_seed = 100
    #olvf_C = 0.1 
    #olvf_Lambda = 50
    #olvf_B = 0.04
    #olvf_option = 1 # 0, 1 or 2
    #stream_mode = "trapezoidal" # or variable, decrease sparsity when variable
    #olsf_C = 0.01
    #olsf_Lambda = 30
    #olsf_B = 0.04
    #olsf_option = 2 # 0, 1 or 2
    #
    #confnet_lr = 0.001
    #confnet_l2 = 0.001
    #signet_lr = 0.0005
    #grad_norm_multiplier = 1
    #new_norm_multiplier = 1
     
#def setParametersWPBC():
    #stream_mode = "trapezoidal" # or variable, decrease sparsity when variable
    #cross_validation_folds = 25
    #random_seed = 1 #41 variable
    #olvf_C = 1 #o.1 variable
    #olvf_C2= 1 #o.1 variable
    #olsf_C = 0.1
    #olsf_Lambda = 30
    #olsf_B = 1
    #olvf_B = 0.08
    #olsf_option = 2 # 0, 1 or 2
    #lr = 0.0001
    #l2 = 0
    #eps =0
    
#def setParametersSVMGuide():
    #phi = 0.01 #0.1 var
    #cross_validation_folds = 20 #20 var
    #random_seed = 1 #20 var
    #olvf_C = 0.1 #0.1 var
    #olvf_C2 = 10 #0.1 var
    #olvf_Lambda = 30 #50 var
    #olvf_B = 1
    #olvf_option = 1 # 0, 1 or 2
    #stream_mode = "trapezoidal" # or variable, decrease sparsity when variable
    #olsf_C = 0.1
    #olsf_Lambda = 30
    #olsf_B = 1
    #olsf_option = 2 # 0, 1 or 2 #sqrt of inconfi
    #lr = 0.0001
    #l2 = 0.01#1
    #eps =0
    
#def setParametersIonosphere():
    #stream_mode = "trapezoidal" # or variable, decrease sparsity when variable
    #cross_validation_folds = 25
    #random_seed = 1 #variable 100
    #olvf_C = 0.01 #variable 0.1
    #olvf_C2 = 0.01 #variable 0.1
    #
    #
    #olvf_option = 1 # 0, 1 or 2
    #olsf_C = 1
    #olsf_Lambda = 30
    #olsf_B = 0.64
    #olvf_B = 1-olsf_B
    #olsf_option = 2 # 0, 1 or 2


     

