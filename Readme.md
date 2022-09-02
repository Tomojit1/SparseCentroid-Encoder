Instruction to use the code package.

This a CPU implementation of Sparse Centroid-Encoder. Scaled Conjugate Gradient Descent (SCG)
is used to update network parameters. The script 'SparseCentroidencoder.py' uses single center
per class where as 'MulticenterSparseCentroidencoder.py' uses multiple centers. After the feature
selection an ANN is used to classifiy the trst samples. The ANN code is written in PyTorch.
Three datasets ALLAML (high dim biological data), COIL20 (image data) and ISOLET (speech data) are
provided with the package. Please uncompress the Data.zip. I used Mac's 'Compress Data' tool. For
the ALLAML data I've provided the three data partitions due to space constraint.
For any question contact Tomojit Ghosh (tomojit.ghosh@colostate.edu).


Requirements:
1. Python: 3.7.4
2. PyTorch: 1.2.0
3. Numpy: 1.17.2
4. ipython: 7.8.0
5. sklearn: 1.0.2


Notes: 
1. Three datasets: ALLAML,COIL20 and ISOLET are given with the package.

2. Seperate script for each data sets:
		ALLAML: testSCE_ALLAML.py
		COIL20: testSCE_COIL20.py
		ISOLET: testSMCE_ISOLET.py

How to run the code:
1. Download the code.

2. Unzip the package. The code was compressed in MacOS BigSur Version 11.6.2.

3. Make sure all the requirements are satisfied.

4. To run the script from ipython use the commands:
    a. ipython
	b. run testSCE_ALLAML.py ==>>for ALLAMML data
	c. run testSCE_COIL20.py ==>>for COIL20 data
	d. run testSMCE_ISOLET.py ==>>for Isolet data
5. To run the script directly from python:
    a. python testSCE_ALLAML.py ==>>for ALLAMML data
    b. python testSCE_COIL20.py ==>>for COIL20 data
    c. python testSMCE_ISOLET.py ==>>for Isolet data
