import numpy as np


   
    
def IDTR_value(is_idtr,  clr_tp):
    clr_IDTR=is_idtr
    clr_IDTR=clr_tp-clr_IDTR


    return clr_IDTR
    
    
def MOTA_plus_value( fn,fp,idsw,idtr_value,tp):
    mp= 1-((fn + fp + idsw + idtr_value) / np.maximum(1.0, tp + fn))


    return mp