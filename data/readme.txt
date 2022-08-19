Overview:

This dataset is associated with the challenge "SurgToolLoc:" happening as part of EndoVis challenge at MICCAI 2022.

The main dataset folder consists of 3 items:

		1. video_clips (folder)
		2. labels.csv (file)
		3. readme.txt (this document)

Video Clips:

All the training video clips for this dataset are under the folder "video_clips". The dataset consists of video clips taken from surgical training exercises using the da Vinci robotic system. During these exercises the surgical trainees are performing standard activities such as dissecting tissue, suturing, etc. 
There are 24,695 video clips, each 30 seconds long and captured at 30fps with a resolution of 720p (1280 x 720) from one channel of the endoscope. For the extent of each clip, there are three tools installed and within the surgical field, although for some clips tools may be obscured or otherwise temporarily not visible. Each clip can contains three of the following 14 possible tools:

	1. 'bipolar dissector''
	2.'bipolar forceps'
	3. 'cadiere forceps'
	4. 'clip applier '
	5. 'force bipolar'
	6. 'grasping retractor'
	7. 'monopolar curved scissors'
	8. 'needle driver'
	9. 'permanent cautery hook/spatula'
	10. 'prograsp forceps'
	11. 'stapler'
	12. 'suction irrigator'
	13. 'tip-up fenestrated grasper'
	14. 'vessel sealer'


Labels:

The labels for each video clip are provided within labels.csv. The set of tools within each clip stay constant, hence there is one label per video clip. Each label is a list of 4 values corresponding to the tool present in the 4 robotic arms ([USM1, USM2, USM3, USM4]). "nan" is used for the camera arm.
