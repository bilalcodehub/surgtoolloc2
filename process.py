# AUTOGENERATED! DO NOT EDIT! File to edit: 09_inference.ipynb.

# %% auto 0
__all__ = ['execute_in_docker', 'VideoLoader', 'UniqueVideoValidator', 'Surgtoolloc_det']

# %% 09_inference.ipynb 3
from utils import *
from fastai.vision.all import *
from skimage.measure import label,regionprops,find_contours
from evalutils import DetectionAlgorithm
from evalutils.validators import UniquePathIndicesValidator,DataFrameValidator
from evalutils.exceptions import ValidationError
import json, random, SimpleITK, gc, cv2
from typing import Tuple, Dict
from pandas import DataFrame
import os

# %% 09_inference.ipynb 5
execute_in_docker = False

# %% 09_inference.ipynb 6
class VideoLoader():
    def load(self, *, fname):
        path = Path(fname)
        print(path)
        if not path.is_file():
            raise IOError(f"Could not load {fname} using {self.__class__.__qualname__}.")
        return [{"path": fname}]

    # only path valid
    def hash_video(self, input_video):
        pass

# %% 09_inference.ipynb 7
class UniqueVideoValidator(DataFrameValidator):
    """
    Validates that each video in the set is unique
    """

    def validate(self, *, df: DataFrame):
        try:
            hashes = df["video"]
        except KeyError:
            raise ValidationError("Column `video` not found in DataFrame.")

        if len(set(hashes)) != len(hashes):
            raise ValidationError("The videos are not unique, please submit a unique video for each case.")

# %% 09_inference.ipynb 9
class Surgtoolloc_det(DetectionAlgorithm):
    def __init__(self):
        super().__init__(
            index_key='input_video',
            file_loaders={'input_video': VideoLoader()},
            input_path=Path("/input/") if execute_in_docker else Path("./test/input/"),
            output_file=Path("/output/surgical-tool-presence.json") if execute_in_docker else Path(
                "./test/output/surgical-tool-presence.json"),
            validators=dict(input_video=(UniquePathIndicesValidator(),)),
        )
        
        # loading ensemble learner
        ensem_path=Path('/opt/algorithm/cls') if execute_in_docker else Path("test/algorithm/cls")
        segmen_path=Path('/opt/algorithm/seg') if execute_in_docker else Path("./test/algorithm/seg")
        self.ensem_learner=[load_learner(m, cpu=False) for m in ensem_path.ls() if m.suffix=='.pkl']
        self.crop_learner=load_learner(segmen_path/'seg_v1.pkl', cpu=False)
        self.codes = ["Background", "Foreground"]

        self.tool_list = ["needle_driver",
                          "monopolar_curved_scissor",
                          "force_bipolar",
                          "clip_applier",
                          "tip_up_fenestrated_grasper",
                          "cadiere_forceps",
                          "bipolar_forceps",
                          "vessel_sealer",
                          "suction_irrigator",
                          "bipolar_dissector",
                          "prograsp_forceps",
                          "stapler",
                          "permanent_cautery_hook_spatula",
                          "grasping_retractor"]


    def crop_images(self, src):
        fs=get_image_files(src)
        preds,_ = self.crop_learner.get_preds(dl=self.crop_learner.dls.test_dl(fs))
        for p, f in zip(preds,self.crop_learner.dl.items):

            fn = f.name

            im=PILImage.create(f)
            (h,w)=im.shape
            mask=PILMask.create((np.array(p.argmax(0))*255).astype(np.uint8))
            mask=Resize((h,w), ResizeMethod.Squish) (mask)

            lbl = label(np.array(mask))
            props = regionprops(lbl)
            x1,y1,x2,y2=props[0].bbox[0],props[0].bbox[2],props[0].bbox[1],props[0].bbox[3]

            im_c = PILImage.create(np.array(im)[x1:y1,x2:y2])
            im_c.save(src/fn)
    
    def extract_images(self, video_file):     
    
        # start the loop
        count = 0
        src=Path(self._input_path)
        
        for i in get_image_files(src): os.remove(i) 
        
        # read the video file    
        cap = cv2.VideoCapture(str(src/video_file))
        
        while True:
            is_read, f = cap.read()
            if not is_read:
                # break out of the loop if there are no frames to read
                break
            name = str(src/f'{count}.jpg')
            cv2.imwrite(name,f)
            count+=1
        cap.release()

    
    def tool_detection_model_output(self):
        
        random_tool_predictions = [random.randint(0, len(self.tool_list) - 1), random.randint(0, len(self.tool_list) - 1)]

        return [self.tool_list[random_tool_predictions[0]], self.tool_list[random_tool_predictions[1]]]

    def tool_detect_json_sample(self):
        # single output dict
        slice_dict = {"slice_nr": 1}
        tool_boolean_dict = {i: False for i in self.tool_list}

        single_output_dict = {**slice_dict, **tool_boolean_dict}

        return single_output_dict

    def process_case(self, *, idx, case):

        # Input video would return the collection of all frames (cap object)
        input_video_file_path = case #VideoLoader.load(case)
        # Detect and score candidates
        scored_candidates = self.predict(case.path) #video file > load evalutils.py

        # return
        # Write resulting candidates to result.json for this case
        return scored_candidates

    def save(self):
        with open(str(self._output_file), "w") as f:
            json.dump(self._case_results[0], f)


    def predict(self, fname) -> Dict:
        """
        Inputs:
        fname -> video file path
        
        Output:
        tools -> list of prediction dictionaries (per frame) in the correct format as described in documentation 
        """
        
        print('Loading, extracting and cropping video file: ' + str(fname))
        self.extract_images(fname)
        self.crop_images(self._input_path)

        fs=get_image_files(self._input_path)
        
        num_frames = len(fs)
        
        ###                                                                     ###
        ###  TODO: adapt the following part for YOUR submission: make prediction
        ###                                                                     ###
        
        print(num_frames)

        # generate output json
        all_frames_predicted_outputs = []
        all_undefined_tools=[]
        
        tta_res=[]
        prs_items=[]
        for learn in self.ensem_learner:
            tta_res.append(learn.tta(dl=learn.dls.test_dl(fs)))
            if len(prs_items)<1:
                prs_items=learn.dl.items

        tta_prs=first(zip(*tta_res))
        tta_prs+=tta_prs[:1]
        tta_prs=torch.stack(tta_prs)

        lbls=[]
        for i in range(len(c)):
            arm_preds = tta_prs[:,:,cfg(i):cfg(i+1)].mean(0);
            arm_idxs = arm_preds.argmax(dim=1)
            arm_vocab = np.array(vocab[i])
            lbls.append(arm_vocab[arm_idxs])

        for usm1,usm2,usm3,usm4,f in zip(lbls[0],lbls[1],lbls[2],lbls[3],prs_items):
            frame_dict=self.tool_detect_json_sample()
            frame_dict['slice_nr']=int(f.stem)
            frame_dict[usm1]=True if usm1 in frame_dict.keys() else all_undefined_tools.append(usm1)
            frame_dict[usm2]=True if usm2 in frame_dict.keys() else all_undefined_tools.append(usm2)
            frame_dict[usm3]=True if usm3 in frame_dict.keys() else all_undefined_tools.append(usm3)
            frame_dict[usm4]=True if usm4 in frame_dict.keys() else all_undefined_tools.append(usm4)
            frame_dict.pop("nan", None)
            frame_dict.pop("blank", None)
            frame_dict.pop("out_of_view", None)
            all_frames_predicted_outputs.append(frame_dict) 

        print(f'List of undefined tools: {set(all_undefined_tools)}.')
        tools=sorted(all_frames_predicted_outputs, key=lambda d: d['slice_nr']) 

        return tools

# %% 09_inference.ipynb 12
if __name__ == "__main__":
    Surgtoolloc_det().process()
