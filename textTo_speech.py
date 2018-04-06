import pyttsx
engine = pyttsx.init()
engine.say('''Images for two specific off-center shifts can be obtained from the left and the right camera. Ad-
ditional shifts between the cameras and all rotations are simulated by viewpoint transformation of
the image from the nearest camera. Precise viewpoint transformation requires 3D scene knowledge
which we don’t have. We therefore approximate the transformation by assuming all points below
the horizon are on flat ground and all points above the horizon are infinitely far away. This works
fine for flat terrain but it introduces distortions for objects that stick above the ground, such as cars,
poles, trees, and buildings. Fortunately these distortions don’t pose a big problem for network train-
ing. The steering label for transformed images is adjusted to one that would steer the vehicle back
to the desired location and orientation in two seconds.
A block diagram of our training system is shown in Figure 2. Images are fed into a CNN which
then computes a proposed steering command. The proposed command is compared to the desired
command for that image and the weights of the CNN are adjusted to bring the CNN output closer to
the desired output. The weight adjustment is accomplished using back propagation as implemented
in the Torch 7 machine learning package.''')
engine.runAndWait()
