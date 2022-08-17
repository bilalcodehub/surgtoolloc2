from fastai.vision.all import *
import gc

# Created datablock first and the used the following query to get all tools
# list(set(sorted(flatten(dls.vocab))))

tools_list = ['stapler',
              'nan',
              'monopolar_curved_scissors',
              'grasping_retractor',
              'needle_driver',
              'cadiere_forceps',
              'force_bipolar',
              'vessel_sealer',
              'tip_up_fenestrated_grasper',
              'bipolar_dissector',
              'clip_applier',
              'permanent_cautery_hook_spatula',
              'bipolar_forceps',
              'suction_irrigator',
              'prograsp_forceps']

images_df=pd.read_csv('data/images_df_final.csv', dtype={'image_id':str,'clip_name':str,'labels':str})

def y1_labeller(i): return re.sub(r"[\[\]]",'',i).split(',')[0].strip()
def y2_labeller(i): return re.sub(r"[\[\]]",'',i).split(',')[1].strip()
def y3_labeller(i): return re.sub(r"[\[\]]",'',i).split(',')[2].strip()
def y4_labeller(i): return re.sub(r"[\[\]]",'',i).split(',')[3].strip()

images_df['y1_label'] = images_df.labels.map(y1_labeller)
images_df['y2_label'] = images_df.labels.map(y2_labeller)
images_df['y3_label'] = images_df.labels.map(y3_labeller)
images_df['y4_label'] = images_df.labels.map(y4_labeller)

def splitter(df):
    train = df.index[~df['valid']].to_list()
    valid = df.index[df['valid']].to_list()
    return train, valid

def get_dblock(item_tfms, batch_tfms):
    
    dblock = DataBlock(
        blocks=(ImageBlock,CategoryBlock(vocab=tools_list),CategoryBlock(vocab=tools_list),CategoryBlock(vocab=tools_list),CategoryBlock(vocab=tools_list)),
        n_inp=1,
        get_x=ColReader('image_id'),
        get_y=[ColReader('y1_label'),ColReader('y2_label'),ColReader('y3_label'),ColReader('y4_label')],
        splitter=splitter,
        item_tfms=item_tfms,
        batch_tfms=batch_tfms)
    
    return dblock

dblock = get_dblock(item_tfms=Resize((180,320), method='squish'), batch_tfms=aug_transforms(size=(180,320), min_scale=1))
dls = dblock.dataloaders(images_df, seed=42, n_workers=8)

def cfg (i): return dls.c[:i].sum()

# defining error rate for each robotic hand tools
def usm1_err(preds,usm1_targs,usm2_targs,usm3_targs,usm4_targs): return error_rate(preds[:,:cfg(1)], usm1_targs)
def usm2_err(preds,usm1_targs,usm2_targs,usm3_targs,usm4_targs): return error_rate(preds[:,cfg(1):cfg(2)], usm2_targs)
def usm3_err(preds,usm1_targs,usm2_targs,usm3_targs,usm4_targs): return error_rate(preds[:,cfg(2):cfg(3)], usm3_targs)
def usm4_err(preds,usm1_targs,usm2_targs,usm3_targs,usm4_targs): return error_rate(preds[:,cfg(3):cfg(4)], usm4_targs)

# defining combined error rate 
def combo_err(preds,usm1_targs,usm2_targs,usm3_targs,usm4_targs): 
    return usm1_err(preds,usm1_targs,usm2_targs,usm3_targs,usm4_targs)+usm2_err(preds,usm1_targs,usm2_targs,usm3_targs,usm4_targs)+usm3_err(preds,usm1_targs,usm2_targs,usm3_targs,usm4_targs)+usm4_err(preds,usm1_targs,usm2_targs,usm3_targs,usm4_targs)

# defining error rate for each robotic hand tools for raw preds from the learner 
def usm1_err_raw(preds,targs): return error_rate(preds[:,:cfg(1)].softmax(dim=1).argmax(dim=1), targs)
def usm2_err_raw(preds,targs): return error_rate(preds[:,cfg(1):cfg(2)].softmax(dim=1).argmax(dim=1), targs)
def usm3_err_raw(preds,targs): return error_rate(preds[:,cfg(2):cfg(3)].softmax(dim=1).argmax(dim=1), targs)
def usm4_err_raw(preds,targs): return error_rate(preds[:,cfg(3):cfg(4)].softmax(dim=1).argmax(dim=1), targs)

# defining loss function for each robotic hand tools
def usm1_loss(preds,usm1_targs,usm2_targs,usm3_targs,usm4_targs,**kwargs): return CrossEntropyLossFlat(reduction='mean')(preds[:,:cfg(1)], usm1_targs,**kwargs)
def usm2_loss(preds,usm1_targs,usm2_targs,usm3_targs,usm4_targs,**kwargs): return CrossEntropyLossFlat(reduction='mean')(preds[:,cfg(1):cfg(2)], usm2_targs,**kwargs)
def usm3_loss(preds,usm1_targs,usm2_targs,usm3_targs,usm4_targs,**kwargs): return CrossEntropyLossFlat(reduction='mean')(preds[:,cfg(2):cfg(3)], usm3_targs,**kwargs)
def usm4_loss(preds,usm1_targs,usm2_targs,usm3_targs,usm4_targs,**kwargs): return CrossEntropyLossFlat(reduction='mean')(preds[:,cfg(3):cfg(4)], usm4_targs,**kwargs)

# defining combined loss
def combo_loss(preds,usm1_targs,usm2_targs,usm3_targs,usm4_targs,**kwargs): 
    return usm1_loss(preds,usm1_targs,usm2_targs,usm3_targs,usm4_targs,**kwargs)+usm2_loss(preds,usm1_targs,usm2_targs,usm3_targs,usm4_targs,**kwargs)+usm3_loss(preds,usm1_targs,usm2_targs,usm3_targs,usm4_targs,**kwargs)+usm4_loss(preds,usm1_targs,usm2_targs,usm3_targs,usm4_targs,**kwargs)

# configuring metrics and loss for learner
metrics_cfg = [usm1_loss,usm2_loss,usm3_loss,usm4_loss,usm1_err,usm2_err,usm3_err,usm4_err, combo_err]

# error rate fns for inference and validation
def usm_err_raw(preds,targs): return error_rate(preds, targs)
def combo_err_raw(preds, targs): 
    return usm_err_raw(preds[:,:cfg(1)].softmax(dim=1),targs[0])+usm_err_raw(preds[:,cfg(1):cfg(2)].softmax(dim=1),targs[1])+usm_err_raw(preds[:,cfg(2):cfg(3)].softmax(dim=1),targs[2])+usm_err_raw(preds[:,cfg(3):cfg(4)].softmax(dim=1),targs[3])
