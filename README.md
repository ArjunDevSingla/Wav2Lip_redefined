# About this Repository:
<i>NOTE: For lipsynced video results, scroll below</i><br><br>

## To Run The Model:
<br> 1. Specify the file paths, and make a new conda environment with Python = 3.6 <br>
<br> 2. Now Install the necessary Liabraries from requirements.txt, and make a media folder, inside of which add the video and audio <br>
<br> 3. Simply Run "Python main.py" and your Final video will be ready

<b>Objectives Achieved:</b>
1. Visual and Audio Quality Lip Sync: The project successfully lip-syncs videos with improved visual and audio quality, ensuring that the lip movements accurately match the spoken words.
2. Robustness for Any Video: Unlike the original Wav2Lip model, the developed AI can handle videos with or without a face in each frame, making it more versatile and error-free.
3. Support for Longer Videos: The model overcomes the limitations of the original Wav2Lip GAN model, now effectively lip-syncing longer videos exceeding 1 minute in duration.
4. Particular Segments Can be extracted easily, unlike the original Model, any part with or without face can be extracted, with te desired audio combined.

<b> Metrics: </b>
I haven't trained the model of any further dataset, so the Metrics is the same for the model
1. Average Mean Squared Error = 5.050382572478908
2. Average Peak Signal to Noise Ratio = 40.32044758489997

<br>
<b>Challenges:</b><br><br>
1. Since, my aim was to extract any particulkar segment, I had to be very concious around timestamps<br>
2. Wav2Lip also doesn't have a mechanism to make a distinction between the target speaker and other faces that appears in the video.<br>
3. The Model does not perform good for high resolution videos<br>
4. Long runtimes for longer videos.<br>
<br><br>

<b>Results include LipSync Videos for:</b>
1. Hindi Voice-Over on English Video.
2. Long Videos with some No-face or Other than Target speaker face with a lot of head and hands movement & Telugu to Hindi Translation Voice-Over synced<br>
<i>results can be reproduced using the colab notebook or can be accessed at this google drive for reference: </i>
https://drive.google.com/file/d/1zjxMi1p3S9SL9UuatoC-2RgWbepyZWUY/view?usp=sharing

<b>NOTE:</b><br>
1. Instead of rendering the whole video at once, my approach breaks it into small pieces, which makes the process faster.<br>
2. I had a different approach of this, rather than just skipping the frames, with no face(Recommende), I wrote the code for extracting the videos with faces and without faces, and individually rendering them. Through this, I didn't made any changes int the internal filing of the model.<br>
3. All my work is done in main.py<br>
```