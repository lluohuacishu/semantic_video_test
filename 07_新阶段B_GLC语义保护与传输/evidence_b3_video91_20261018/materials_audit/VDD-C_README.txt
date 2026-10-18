This readme.txt file was generated on 4/16/2021 by Michael Fulton

-------------------
GENERAL INFORMATION
-------------------
1. Video Diver Dataset (VDD-C, the C stands for the 100,000 images in the dataset.)


2. 
  Principal Investigator Contact Information
        Name: Junaed Sattar
           Institution: University of Minnesota
           Email: junaed@umn.edu
	   ORCID:0000-0002-3983-6265

  Associate or Co-investigator Contact Information
        Name: Karin de Langis
           Institution: University of Minnesota
           Email:dento019@umn.edu
	   ORCID: N/A

  Associate or Co-investigator Contact Information
           Name:Michael Fulton
           Institution: University of Minnesota
           Email: fulto081@umn.edu
	   ORCID:0000-0003-1842-6045

3. Date of data collection: 
	Between 2017 and 2020.
	
4. Geographic location of data collection: 
	Data was collected near Holetown, Barbados, and in University of Minnesota pools in Minneapolis, MN. 

5. Information about funding sources that supported the collection of the data:
	This work was funded by the Minnesota Robotics Institute and the National Science Foundation.

--------------------------
SHARING/ACCESS INFORMATION
-------------------------- 


1. Licenses/restrictions placed on the data: 
	This data is under a Creative Commons license, please see the LICENSE.rdf file for more specific information.

	All persons in this dataset were contacted to confirm their consent to be included in the dataset. These written affirmations of consent are stored.

2. Links to publications that cite or use the data:
	https://arxiv.org/abs/2012.05701
		

3. Links to other publicly accessible locations of the data:
	N/A

4. Links/relationships to ancillary data sets:
	N/A

5. Was data derived from another source?
	No.


6. Recommended citation for the data:

MLA: de Langis, Karin, Michael Fulton, and Junaed Sattar. "An Analysis of Deep Object Detectors For Diver Detection." arXiv preprint arXiv:2012.05701 (2020).
APA: de Langis, K., Fulton, M., & Sattar, J. (2020). An Analysis of Deep Object Detectors For Diver Detection. arXiv preprint arXiv:2012.05701.
Chicago: de Langis, Karin, Michael Fulton, and Junaed Sattar. "An Analysis of Deep Object Detectors For Diver Detection." arXiv preprint arXiv:2012.05701 (2020).
Harvard: de Langis, K., Fulton, M. and Sattar, J., 2020. An Analysis of Deep Object Detectors For Diver Detection. arXiv preprint arXiv:2012.05701.
Vancouver: de Langis K, Fulton M, Sattar J. An Analysis of Deep Object Detectors For Diver Detection. arXiv preprint arXiv:2012.05701. 2020 Nov 25.

BibTEX form:
@misc{delangis2020analysis,
      title={An Analysis of Deep Object Detectors For Diver Detection}, 
      author={Karin de Langis and Michael Fulton and Junaed Sattar},
      year={2020},
      eprint={2012.05701},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

---------------------
DATA & FILE OVERVIEW
---------------------
1. File List
   A. Filename: images.zip
      Short description:  This compressed folder contains the images of the VDD-C dataset in jpg format.
        
   B. Filename: yolo_labels.zip
      Short description:  This compressed folder contains the annotation labels of the VDD-C dataset in the YOLO label format.
  
   C. Filename: voc_labels.zip
      Short description:  This compressed folder contains the annotation labels of the VDD-C dataset in the YOLO label format.
 
   D. Filename: tfrecord_labels.zip
      Short description:  This compressed folder contains the annotation labels of the VDD-C dataset in the tfrecord label format.

   E. Filename: tfsequence_labels_test-set.zip
      Short description:  This compressed folder contains the annotation labels of the VDD-C dataset in the tfsequence label format. Due to the size of the dataset, this compressed folder contains only the test set.

   F. Filename: tfsequence_labels_val-set.zip
      Short description:  This compressed folder contains the annotation labels of the VDD-C dataset in the tfsequence label format. Due to the size of the dataset, this compressed folder contains only the validation set.

   G. Filename: tfsequence_labels_train-set-#.zip.part
      Short description:  This compressed folder contains the annotation labels of the VDD-C dataset in the tfsequence label format. Due to the size of the dataset, the training set is split into 8 zip folder parts. In order to unzip these, you must join them first.


2. Relationship between files:        
	The images.zip contains image data, which is required for the labels, for whatever label format you are using.

	To unzip the tfsequence_labels_train-set#.zip.part files, please use the following approach on:
	Linux: tfsequence_labels_train-set* | tar xz
	Windows:copy /b  tfsequence_labels_train-set-1.zip.part + tfsequence_labels_train-set-2.zip.part + tfsequence_labels_train-set-3.zip.part + tfsequence_labels_train-set-14zip.part + tfsequence_labels_train-set-5.zip.part + tfsequence_labels_train-set-6.zip.part + tfsequence_labels_train-set-7.zip.part tfsequence_labels_train-set.zip
		then unzip.
	Source of this approach: https://stackoverflow.com/questions/1120095/split-files-using-tar-gz-zip-or-bzip2


3. Additional related data collected that was not included in the current data package:
	The original data that this dataset was created from can be provided to those interested, but due to its size, has not been uploaded. The original data differs in that it has certain 
	frames which were eliminated due to blur, significant occlusion of divers, or above-water data. There are also original labels, which have some malformed labels with invalid bounding 
	box coordinates, etc. These were eliminated using a script, which can also be provided to anyone interested. 

4. Are there multiple versions of the dataset? yes/no
   No

--------------------------
METHODOLOGICAL INFORMATION
--------------------------
1. Description of methods used for collection/generation of data: 
	This data was collected from field trials with robots and pool experiments with robots. All of these videos were collected on handheld GoPro devices.

2. Methods for processing the data: <describe how the submitted data were generated from the raw or collected data>
	The videos were first split into smaller portions, then annotated using the EVA labeling tool. Following this,
	sections of each video which were too blury or occluded were eliminated, leaving us with the final data.

3. Instrument- or software-specific information needed to interpret the data:
	N/A


4. Standards and calibration information, if appropriate:
	N/A

5. Environmental/experimental conditions:
	N/A

6. Describe any quality-assurance procedures performed on the data:
	Every video was subjected to a proofreading process after annotation, where the annotations were drawn over each frame of the video, 
	then watched through by a a group of volunteers for mistakes and issues, which were then fixed.

7. People involved with sample collection, processing, analysis and/or submission:
	The authors did the processing and analysis, and those involved in sample collection are anonymous for their security.